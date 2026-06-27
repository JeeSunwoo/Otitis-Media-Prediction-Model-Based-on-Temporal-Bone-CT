"""
train.py  —  v40 otitis 재학습 (stratified fold + aug 강화 + dropout↑)
============================================================
[v39 → v40 변경]  fold 분할을 Rt otitis 양성 여부로 stratify
  - 기존: 환자 ID 랜덤 셔플 → fold 마다 Rt 양성 환자 수가 2~5명으로 들쭉날쭉
    → fold-val Rt F1 편차 심함(0.33~0.50), best 가 운인지 실력인지 판단 어려움.
  - 변경: Rt otitis 양성 보유 환자 / 미보유 환자를 각각 fold 에 라운드로빈 분배
    → 모든 fold 가 비슷한 Rt 양성 비율 → fold 편차↓, best 판단 신뢰도↑.
  - ★ 효과는 '안정화/판단 명확성'이 주. Rt 양성 절대수(≈19명)는 불변이므로
    성능 점프보다는 일관성 개선이 목적. 떨어지지 않고 약간 낫거나 비슷 기대.
  - LOFO(recompute_thr)는 v44 에서 효과 없어 이번엔 미사용. 앙상블 thr 산출만 사용.
[v39 유지] aug 강화(±13/7%/0.75-1.25), otitis dropout↑(model.py),
           A(alpha 0.5+clamp), B(per-side loss), C(otitis best=per-side F1),
           앙상블 thr 산출, cand 마스크 정합.
============================================================
"""
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score
from tqdm import tqdm

import model as M
from model import get_temporal_model, get_otitis_model

# ----------------------------- 공통 하이퍼파라미터 -----------------------------
IMAGE_SIZE = M.IMAGE_SIZE
PATIENTS_PER_BATCH = 2
NUM_EPOCHS = 50
PATIENCE = 12
FREEZE_EPOCHS = 8
WARMUP_EPOCHS = 3
HEAD_LR = 2.5e-5
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-3
GAMMA = 2.0
SEED = 42
K_FOLDS = 5

# ----------------------------- stage 별 설정 -----------------------------
TEMPORAL_BACKBONE = "efficientnet_b0"
OTITIS_BACKBONE = "efficientnet_b0"
USE_DIFF = True
OTITIS_USE_ROI = False
OTITIS_ROI_BOX = M.DEFAULT_OTITIS_ROI

USE_LR_SWAP = False
LR_SWAP_PROB = 0.5
LAMBDA_AUX = 0.5

USE_PER_SIDE_LOSS = True
LT_SIDE_WEIGHT = 0.8

USE_AMP = True and torch.cuda.is_available()

CAND_MARGIN = 5
HARD_NEG_FRAC = 0.3
POS_WEIGHT_CAP = 8.0

OTITIS_SWEEP_USE_CAND = True

# ★ v40: fold stratify 기준 (otitis stage 에서만 사용)
STRATIFY_OTITIS = True

TEMPORAL_PP = {"median_k": 3, "max_gap": 2, "min_run": 2, "keep_largest": False, "dilation_margin": 5}
OTITIS_PP = {"min_run": 2, "max_gap": 2, "temporal_gate_margin": 5}

torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
VAL_TF = M.build_eval_tf()
_RESIZE = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
_TENSOR_NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize(M.MEAN, M.STD)])


# ----------------------------- 좌우/뷰 공유 증강 -----------------------------
def _rand_geo_photo():
    # v39: aug 적당히 강화 (외우기 방지). 회전 ±13, 이동 7%, 밝기/대비 0.75~1.25.
    return {"angle": random.uniform(-13, 13),
            "tx": random.uniform(-0.07 * IMAGE_SIZE, 0.07 * IMAGE_SIZE),
            "ty": random.uniform(-0.07 * IMAGE_SIZE, 0.07 * IMAGE_SIZE),
            "b": random.uniform(0.75, 1.25), "c": random.uniform(0.75, 1.25),
            "sigma": random.uniform(0.1, 1.0)}


def _apply_geo_photo(x, p):
    x = TF.affine(x, angle=p["angle"], translate=(p["tx"], p["ty"]), scale=1.0, shear=[0.0, 0.0])
    x = TF.adjust_brightness(x, p["b"]); x = TF.adjust_contrast(x, p["c"])
    return TF.gaussian_blur(x, kernel_size=3, sigma=p["sigma"])


def make_views(rt_half, lt_half, is_train, use_roi, roi_box):
    p = _rand_geo_photo() if is_train else None
    rt_f, lt_f = _RESIZE(rt_half), _RESIZE(lt_half)
    if is_train:
        rt_f, lt_f = _apply_geo_photo(rt_f, p), _apply_geo_photo(lt_f, p)
    full = torch.stack([_TENSOR_NORM(rt_f), _TENSOR_NORM(lt_f)], 0)
    if not use_roi:
        return full, None
    rt_r, lt_r = _RESIZE(M.roi_crop(rt_half, roi_box)), _RESIZE(M.roi_crop(lt_half, roi_box))
    if is_train:
        rt_r, lt_r = _apply_geo_photo(rt_r, p), _apply_geo_photo(lt_r, p)
    roi = torch.stack([_TENSOR_NORM(rt_r), _TENSOR_NORM(lt_r)], 0)
    return full, roi


# ----------------------------- 샘플 빌드 -----------------------------
def build_samples(csv_path, data_root, stage):
    df = pd.read_csv(csv_path).drop_duplicates(subset=["No", "R/L", "Image number"], keep="first")
    slice_cols = [str(i) for i in range(1, 133)]
    target_im = "temporal area" if stage == "temporal" else "otitis media"
    LABEL = [("Rt", target_im), ("Lt", target_im)]
    TEMP = [("Rt", "temporal area"), ("Lt", "temporal area")]

    patients = []
    for p_id_int, g in tqdm(df.groupby("No"), desc=f"{stage} 샘플 빌드"):
        p_id = str(int(p_id_int))
        img_dir = M.find_patient_dcm_dir(data_root, p_id, subdirs=("", "train", "val"))
        if img_dir is None:
            continue
        lrows = [g[(g["R/L"] == rl) & (g["Image number"] == im)] for rl, im in LABEL]
        lrows = [r.iloc[0] if len(r) else None for r in lrows]
        trows = [g[(g["R/L"] == rl) & (g["Image number"] == im)] for rl, im in TEMP]
        trows = [r.iloc[0] if len(r) else None for r in trows]

        slices = []
        for col in slice_cols:
            s = int(col)
            path = os.path.join(img_dir, f"{s:04d}.dcm")
            if not os.path.exists(path):
                continue
            label, mask, tpos = [0.0, 0.0], [0.0, 0.0], [0, 0]
            for c, row in enumerate(lrows):
                if row is not None and not pd.isna(row[col]):
                    label[c] = float(int(row[col])); mask[c] = 1.0
            for c, row in enumerate(trows):
                if row is not None and not pd.isna(row[col]) and int(row[col]) == 1:
                    tpos[c] = 1
            if sum(mask) == 0:
                continue
            slices.append({"path": path, "label": label, "mask": mask, "tpos": tpos,
                           "slice_idx": s, "slice_norm": s / 132.0})
        if not slices:
            continue
        if stage == "otitis":
            for c in range(2):
                tarr = np.array([sl["tpos"][c] for sl in slices], dtype=int)
                cand = M.dilate_1d_mask(tarr, CAND_MARGIN)
                for i, sl in enumerate(slices):
                    sl.setdefault("cand", [0, 0])[c] = int(cand[i])
        patients.append({"p_id": p_id, "slices": slices})
    return patients


# ★ v40: 환자의 Rt otitis 양성 슬라이스 보유 여부 (stratify 기준)
def rt_otitis_positive(patient):
    """side0=Rt. 라벨 1 & 마스크 1 인 슬라이스가 하나라도 있으면 양성 환자."""
    for s in patient["slices"]:
        if s["label"][0] == 1.0 and s["mask"][0] == 1.0:
            return True
    return False


def make_folds(patients, stage):
    """fold 분할. otitis + STRATIFY_OTITIS 면 Rt 양성 여부로 stratify,
    아니면 기존 랜덤 셔플. SEED 고정으로 재현 가능."""
    pids = [p["p_id"] for p in patients]
    if stage == "otitis" and STRATIFY_OTITIS:
        pos_pids = [p["p_id"] for p in patients if rt_otitis_positive(p)]
        neg_pids = [p["p_id"] for p in patients if not rt_otitis_positive(p)]
        np.random.seed(SEED)
        np.random.shuffle(pos_pids); np.random.shuffle(neg_pids)
        folds = [set() for _ in range(K_FOLDS)]
        # 양성/음성을 각각 라운드로빈 → 모든 fold 가 비슷한 양성 비율
        for i, pid in enumerate(pos_pids):
            folds[i % K_FOLDS].add(pid)
        for i, pid in enumerate(neg_pids):
            folds[i % K_FOLDS].add(pid)
        print(f"  [stratified fold] Rt otitis 양성 환자 {len(pos_pids)}명 / 음성 {len(neg_pids)}명")
        for k, f in enumerate(folds):
            n_pos = sum(1 for pid in f if pid in set(pos_pids))
            print(f"    fold {k+1}: 총 {len(f)}명 (Rt양성 {n_pos}명)")
    else:
        np.random.seed(SEED); np.random.shuffle(pids)
        folds = [set(pids[i::K_FOLDS]) for i in range(K_FOLDS)]
    return folds


class SeqDataset(Dataset):
    def __init__(self, patients, stage, is_train, use_roi, roi_box):
        self.patients = patients; self.stage = stage; self.is_train = is_train
        self.use_roi = use_roi; self.roi_box = roi_box

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        slices = self.patients[idx]["slices"]
        do_swap = (self.is_train and USE_LR_SWAP and self.stage == "otitis"
                   and random.random() < LR_SWAP_PROB)
        fulls, rois, metas, poss, labels, masks, cands = [], [], [], [], [], [], []
        for s in slices:
            image = M.dcm_to_multiwindow_rgb(s["path"])
            rt, lt = M.split_lr(image)
            if do_swap:
                rt, lt = lt, rt
            full, roi = make_views(rt, lt, self.is_train, self.use_roi, self.roi_box)
            fulls.append(full)
            if roi is not None:
                rois.append(roi)
            metas.append(torch.tensor([s["slice_norm"]], dtype=torch.float))
            poss.append(s["slice_idx"])
            lab, msk, cnd = s["label"], s["mask"], s.get("cand", [1, 1])
            if do_swap:
                lab = [lab[1], lab[0]]
                msk = [msk[1], msk[0]]
                cnd = [cnd[1], cnd[0]]
            labels.append(torch.tensor(lab, dtype=torch.float))
            masks.append(torch.tensor(msk, dtype=torch.float))
            cands.append(torch.tensor(cnd, dtype=torch.float))
        out = {"full": torch.stack(fulls, 0), "meta": torch.stack(metas, 0),
               "pos": torch.tensor(poss, dtype=torch.long),
               "label": torch.stack(labels, 0), "mask": torch.stack(masks, 0),
               "cand": torch.stack(cands, 0),
               "roi": torch.stack(rois, 0) if rois else None}
        return out


def collate_fn(batch):
    B = len(batch); maxS = max(b["full"].shape[0] for b in batch)
    C, H, W = batch[0]["full"].shape[2:]
    has_roi = batch[0]["roi"] is not None
    full = torch.zeros(B, maxS, 2, C, H, W)
    roi = torch.zeros(B, maxS, 2, C, H, W) if has_roi else None
    meta = torch.zeros(B, maxS, 1); pos = torch.zeros(B, maxS, dtype=torch.long)
    label = torch.zeros(B, maxS, 2); mask = torch.zeros(B, maxS, 2); cand = torch.zeros(B, maxS, 2)
    pad = torch.ones(B, maxS, dtype=torch.bool)
    for i, b in enumerate(batch):
        S = b["full"].shape[0]
        full[i, :S] = b["full"]
        if has_roi:
            roi[i, :S] = b["roi"]
        meta[i, :S] = b["meta"]; pos[i, :S] = b["pos"]
        label[i, :S] = b["label"]; mask[i, :S] = b["mask"]; cand[i, :S] = b["cand"]
        pad[i, :S] = False
    return full, roi, meta, pos, pad, label, mask, cand


# ----------------------------- 손실 / 지표 -----------------------------
def _focal_bce_elementwise(logits, targets, pos_weight, gamma=GAMMA):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    pt = torch.exp(-bce)
    return (1 - pt) ** gamma * bce


def masked_focal_bce(logits, targets, mask, pos_weight, gamma=GAMMA):
    fl = _focal_bce_elementwise(logits, targets, pos_weight, gamma)
    return (fl * mask).sum() / mask.sum().clamp(min=1.0)


def per_side_focal_bce(logits, targets, mask, pos_weight, lt_weight, gamma=GAMMA):
    fl = _focal_bce_elementwise(logits, targets, pos_weight, gamma)
    fm = fl * mask
    rt_loss = fm[:, :, 0].sum() / mask[:, :, 0].sum().clamp(min=1.0)
    lt_loss = fm[:, :, 1].sum() / mask[:, :, 1].sum().clamp(min=1.0)
    return rt_loss + lt_weight * lt_loss


def pos_weight_for(patients, device, stage):
    labs = np.array([s["label"] for p in patients for s in p["slices"]])
    msk = np.array([s["mask"] for p in patients for s in p["slices"]])
    pw = torch.zeros(2)
    for c in range(2):
        valid = msk[:, c] == 1
        n_pos = int(labs[valid, c].sum()); n_neg = int(valid.sum()) - n_pos
        ratio = n_neg / max(n_pos, 1)
        pw[c] = min(math.sqrt(ratio), POS_WEIGHT_CAP) if stage == "otitis" else ratio
    return pw.to(device)


def evaluate_collect(model, loader, device):
    model.eval(); P, L, Msk = [], [], []
    with torch.no_grad():
        for full, roi, meta, pos, pad, label, mask, cand in loader:
            full, meta, pos, pad = full.to(device), meta.to(device), pos.to(device), pad.to(device)
            roi_d = roi.to(device) if roi is not None else None
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(full, meta, pos, pad) if roi_d is None else model(full, meta, pos, pad, roi=roi_d)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits).float().cpu(); pad_c = pad.cpu()
            P.append(probs.reshape(-1, 2)); L.append(label.reshape(-1, 2))
            Msk.append((mask * (~pad_c).unsqueeze(-1).float()).reshape(-1, 2))
    return torch.cat(P).numpy(), torch.cat(L).numpy(), torch.cat(Msk).numpy()


def evaluate_collect_ensemble(models, patients, device, stage, use_cand=False):
    use_roi = (stage == "otitis" and OTITIS_USE_ROI)
    ds = SeqDataset(patients, stage, False, use_roi, OTITIS_ROI_BOX)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=collate_fn)
    for m in models:
        m.eval()
    P, L, Msk = [], [], []
    with torch.no_grad():
        for full, roi, meta, pos, pad, label, mask, cand in loader:
            full, meta, pos, pad = full.to(device), meta.to(device), pos.to(device), pad.to(device)
            roi_d = roi.to(device) if roi is not None else None
            s = None
            for m in models:
                with torch.cuda.amp.autocast(enabled=USE_AMP):
                    out = m(full, meta, pos, pad) if roi_d is None else m(full, meta, pos, pad, roi=roi_d)
                lg = (out[0] if isinstance(out, tuple) else out).float()
                s = lg if s is None else s + lg
            probs = torch.sigmoid(s / len(models)).cpu()
            pad_c = pad.cpu()
            eval_mask = mask * (~pad_c).unsqueeze(-1).float()
            if use_cand and stage == "otitis":
                eval_mask = eval_mask * cand
            P.append(probs.reshape(-1, 2)); L.append(label.reshape(-1, 2))
            Msk.append(eval_mask.reshape(-1, 2))
    return torch.cat(P).numpy(), torch.cat(L).numpy(), torch.cat(Msk).numpy()


def sweep_thresholds(P, L, Msk, grid):
    best_thrs = np.full(2, 0.5); class_f1s = [0.0] * 2
    for c in range(2):
        valid = Msk[:, c] == 1
        if valid.sum() == 0:
            continue
        yt, yp = L[valid, c].astype(int), P[valid, c]
        bf, bt = -1.0, 0.5
        for thr in grid:
            f1c = f1_score(yt, (yp >= thr).astype(int), average="macro", zero_division=0)
            if f1c > bf:
                bf, bt = f1c, thr
        best_thrs[c], class_f1s[c] = bt, bf
    return best_thrs, class_f1s


def pooled_f1(P, L, Msk, thrs):
    yt_pool, yp_pool = [], []
    for c in range(2):
        valid = Msk[:, c] == 1
        if valid.sum() == 0:
            continue
        yt_pool.append(L[valid, c].astype(int)); yp_pool.append((P[valid, c] >= thrs[c]).astype(int))
    if not yt_pool:
        return 0.0
    return f1_score(np.concatenate(yt_pool), np.concatenate(yp_pool), average="macro", zero_division=0)


def per_side_pos_f1(P, L, Msk, thrs):
    f1s = []
    for c in range(2):
        valid = Msk[:, c] == 1
        if valid.sum() == 0:
            f1s.append(0.0); continue
        yt = L[valid, c].astype(int); yp = (P[valid, c] >= thrs[c]).astype(int)
        f1s.append(f1_score(yt, yp, average="binary", pos_label=1, zero_division=0))
    return float(np.mean(f1s)), f1s


def fn_fp_report(P, L, Msk, thrs, names):
    for c in range(2):
        valid = Msk[:, c] == 1
        if valid.sum() == 0:
            continue
        yt = L[valid, c].astype(int); yp = (P[valid, c] >= thrs[c]).astype(int)
        tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"    {names[c]:12s} TP={tp:4d} FP={fp:4d} FN={fn:4d}  F1={f1:.4f} @thr={thrs[c]:.2f}")


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ----------------------------- fold 학습 -----------------------------
def make_model(stage, device):
    if stage == "temporal":
        return get_temporal_model(TEMPORAL_BACKBONE, use_diff=USE_DIFF).to(device)
    return get_otitis_model(OTITIS_BACKBONE, use_roi=OTITIS_USE_ROI, use_diff=USE_DIFF).to(device)


def train_one_fold(stage, fold_idx, train_pat, val_pat, device, thr_grid):
    names = ["Rt_temporal", "Lt_temporal"] if stage == "temporal" else ["Rt_otitis", "Lt_otitis"]
    use_roi = (stage == "otitis" and OTITIS_USE_ROI)
    train_ds = SeqDataset(train_pat, stage, True, use_roi, OTITIS_ROI_BOX)
    val_ds = SeqDataset(val_pat, stage, False, use_roi, OTITIS_ROI_BOX)
    train_loader = DataLoader(train_ds, batch_size=PATIENTS_PER_BATCH, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    pos_weight = pos_weight_for(train_pat, device, stage)
    model = make_model(stage, device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    bb_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    optimizer = optim.AdamW([{"params": bb_params, "lr": BACKBONE_LR},
                             {"params": head_params, "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    def lr_factor(epoch):
        if epoch < WARMUP_EPOCHS:
            return 0.1 + 0.9 * (epoch + 1) / WARMUP_EPOCHS
        t = (epoch - WARMUP_EPOCHS) / max(1, NUM_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * t))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)

    best_f1, best_state, patience = 0.0, None, 0
    for epoch in range(NUM_EPOCHS):
        if epoch == FREEZE_EPOCHS:
            for p in model.backbone.parameters():
                p.requires_grad = True
            print("  [2단계] Backbone unfreeze")
        model.train(); running = 0.0
        print(f"\n  [{stage} | Fold {fold_idx+1} | Epoch {epoch+1}/{NUM_EPOCHS}] "
              f"bb_lr={optimizer.param_groups[0]['lr']:.2e} head_lr={optimizer.param_groups[1]['lr']:.2e}")
        for full, roi, meta, pos, pad, label, mask, cand in tqdm(train_loader, unit="pat", leave=False):
            full, meta, pos = full.to(device), meta.to(device), pos.to(device)
            pad, label, mask, cand = pad.to(device), label.to(device), mask.to(device), cand.to(device)
            roi_d = roi.to(device) if roi is not None else None
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                out = model(full, meta, pos, pad) if roi_d is None else model(full, meta, pos, pad, roi=roi_d)
                valid = (~pad).unsqueeze(-1).float()
                if stage == "otitis":
                    logits, d = out
                    hard_neg = (torch.rand_like(cand) < HARD_NEG_FRAC).float() * (1.0 - cand)
                    train_mask = mask * valid * torch.clamp(cand + hard_neg, max=1.0)
                    if USE_PER_SIDE_LOSS:
                        loss = per_side_focal_bce(logits, label, train_mask, pos_weight, LT_SIDE_WEIGHT)
                    else:
                        loss = masked_focal_bce(logits, label, train_mask, pos_weight)
                    d_target = label[:, :, 0] - label[:, :, 1]
                    aux_mask = mask[:, :, 0] * mask[:, :, 1] * valid.squeeze(-1)
                    aux = F.smooth_l1_loss(d, d_target, reduction="none")
                    aux_loss = (aux * aux_mask).sum() / aux_mask.sum().clamp(min=1.0)
                    loss = loss + LAMBDA_AUX * aux_loss
                else:
                    logits = out
                    train_mask = mask * valid
                    loss = masked_focal_bce(logits, label, train_mask, pos_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        train_loss = running / max(len(train_loader), 1)

        P, L, Msk = evaluate_collect(model, val_loader, device)
        best_thrs, class_f1s = sweep_thresholds(P, L, Msk, thr_grid)
        if stage == "otitis":
            sel_f1, side_f1s = per_side_pos_f1(P, L, Msk, best_thrs)
            disp = f"Rt:{side_f1s[0]:.3f}@{best_thrs[0]:.2f} Lt:{side_f1s[1]:.3f}@{best_thrs[1]:.2f}  (sel={sel_f1:.4f})"
        else:
            sel_f1 = pooled_f1(P, L, Msk, best_thrs)
            disp = "  ".join(f"{names[c]}:{class_f1s[c]:.3f}@{best_thrs[c]:.2f}" for c in range(2)) + f"  (sel={sel_f1:.4f})"
        print(f"  Train Loss: {train_loss:.4f}\n  {disp}")
        scheduler.step()
        if sel_f1 > best_f1:
            best_f1, best_state, patience = sel_f1, cpu_state(model), 0
            if stage == "otitis" and "alpha" in best_state:
                print(f"  ★ best 갱신 ({best_f1:.4f})  alpha={float(best_state['alpha'].reshape(-1)[0]):.3f}")
            else:
                print(f"  ★ best 갱신 ({best_f1:.4f})")
        else:
            patience += 1
            print(f"  미개선 ({patience}/{PATIENCE})")
            if patience >= PATIENCE:
                print("  [Early Stopping]"); break

    return best_f1, best_state


# ----------------------------- 체크포인트 통합 -----------------------------
def _merge_unified(temporal_path="temporal_model.pth", otitis_path="otitis_model.pth",
                   out_path="best_model.pth"):
    if not (os.path.exists(temporal_path) and os.path.exists(otitis_path)):
        return False

    def _load(p):
        try:
            return torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(p, map_location="cpu")

    t, o = _load(temporal_path), _load(otitis_path)
    torch.save({"format": "unified_staged_v40", "temporal": t, "otitis": o}, out_path)
    return True


# ----------------------------- stage 학습 엔트리 -----------------------------
def train_stage(stage):
    assert stage in ("temporal", "otitis")
    DATA_ROOT = M.find_data_root()
    TRAIN_CSV = M.find_file("train_set.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bb = TEMPORAL_BACKBONE if stage == "temporal" else OTITIS_BACKBONE
    use_roi = (stage == "otitis" and OTITIS_USE_ROI)
    names = ["Rt_temporal", "Lt_temporal"] if stage == "temporal" else ["Rt_otitis", "Lt_otitis"]
    thr_grid = np.arange(0.20, 0.91, 0.02) if stage == "otitis" else np.arange(0.20, 0.81, 0.02)
    ps = (USE_PER_SIDE_LOSS and stage == "otitis")
    strat = (STRATIFY_OTITIS and stage == "otitis")
    print(f"\n{'#'*64}\n#  STAGE = {stage} | backbone={bb} | per_side_loss={ps}(lt_w={LT_SIDE_WEIGHT}) "
          f"| aug=strong | stratify={strat} | AMP={USE_AMP}\n{'#'*64}")
    print(f"  DATA_ROOT={DATA_ROOT} | HU_WINDOWS={M.HU_WINDOWS}")

    patients = build_samples(TRAIN_CSV, DATA_ROOT, stage)
    if not patients:
        raise RuntimeError(f"환자 0명 — DATA_ROOT({DATA_ROOT})/CSV 확인")
    n_slices = sum(len(p["slices"]) for p in patients)
    print(f"환자 {len(patients)}명 / 슬라이스 {n_slices}개")

    folds = make_folds(patients, stage)

    fold_states, fold_scores = [], []
    for k in range(K_FOLDS):
        val_p = folds[k]
        tr = [p for p in patients if p["p_id"] not in val_p]
        va = [p for p in patients if p["p_id"] in val_p]
        print(f"\n##### [{stage}] FOLD {k+1}/{K_FOLDS}  train={len(tr)} val={len(va)} #####")
        score, state = train_one_fold(stage, k, tr, va, device, thr_grid)
        fold_states.append(state); fold_scores.append(score)

    print(f"\n  [thr 산출] 5-fold 앙상블(logit평균)로 train 전체 재예측 → sweep "
          f"(cand={OTITIS_SWEEP_USE_CAND and stage=='otitis'})")
    ens_models = []
    for st in fold_states:
        m = make_model(stage, device); m.load_state_dict(st); m.eval()
        ens_models.append(m)
    use_cand = (OTITIS_SWEEP_USE_CAND and stage == "otitis")
    EP, EL, EM = evaluate_collect_ensemble(ens_models, patients, device, stage, use_cand=use_cand)
    best_thrs, class_f1s = sweep_thresholds(EP, EL, EM, thr_grid)
    if stage == "otitis":
        ens_sel, ens_sides = per_side_pos_f1(EP, EL, EM, best_thrs)
    else:
        ens_sel = pooled_f1(EP, EL, EM, best_thrs); ens_sides = None
    del ens_models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fold_alphas = None
    if stage == "temporal":
        ckpt = {"model_type": "temporal", "folds": fold_states, "backbone": bb,
                "use_diff": USE_DIFF, "thresholds": best_thrs.tolist(),
                "hu_windows": M.HU_WINDOWS, "left_half_is_rt": M.LEFT_HALF_IS_RT,
                "image_size": IMAGE_SIZE, "preprocess_version": M.PREPROCESS_VERSION,
                "postprocess": TEMPORAL_PP, "thr_source": "ensemble_logitmean_train"}
        path = "temporal_model.pth"
    else:
        fold_alphas = [float(s["alpha"].reshape(-1)[0]) for s in fold_states if "alpha" in s]
        ckpt = {"model_type": "otitis", "folds": fold_states, "backbone": bb,
                "use_diff": USE_DIFF, "use_roi": OTITIS_USE_ROI, "roi_box": list(OTITIS_ROI_BOX),
                "thresholds": best_thrs.tolist(), "hu_windows": M.HU_WINDOWS,
                "left_half_is_rt": M.LEFT_HALF_IS_RT, "image_size": IMAGE_SIZE,
                "preprocess_version": M.PREPROCESS_VERSION, "postprocess": OTITIS_PP,
                "lr_swap": bool(USE_LR_SWAP), "lr_swap_prob": LR_SWAP_PROB,
                "alpha": fold_alphas, "lambda_aux": LAMBDA_AUX, "diff_branch": True,
                "per_side_loss": bool(USE_PER_SIDE_LOSS), "lt_side_weight": LT_SIDE_WEIGHT,
                "thr_source": "ensemble_logitmean_train", "sweep_use_cand": use_cand,
                "aug": "strong_v39", "stratify": bool(STRATIFY_OTITIS)}
        path = "otitis_model.pth"
    torch.save(ckpt, path)

    print(f"\n{'='*64}")
    print(f"  [{stage}] Fold별 best(sel) F1 : {[round(float(s),4) for s in fold_scores]}  평균 {np.mean(fold_scores):.4f}")
    print(f"  [앙상블 thr 산출] train 전체 sel F1 : {ens_sel:.4f}"
          + (f"  (Rt:{ens_sides[0]:.3f} Lt:{ens_sides[1]:.3f})" if ens_sides else ""))
    fn_fp_report(EP, EL, EM, best_thrs, names)
    print(f"  THRESHOLD : {[round(float(t),2) for t in best_thrs]}  [{names[0]}, {names[1]}]")
    if fold_alphas is not None:
        print(f"  learned alpha (fold별) : {[round(a,4) for a in fold_alphas]}  평균 {np.mean(fold_alphas):.4f}")
    print(f"  saved -> {path}")

    if _merge_unified():
        print(f"  unified -> best_model.pth (temporal + otitis)")
    print(f"{'='*64}")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = [arg] if arg in ("temporal", "otitis") else ["temporal", "otitis"]
    for st in stages:
        train_stage(st)