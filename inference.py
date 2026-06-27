"""
inference.py  —  v37 단계형 추론 (앙상블=logit평균) + threshold 수동/자동 스위치
============================================================
[threshold 스위치]
  THR_MODE = "auto"   : ckpt 에 저장된 threshold 사용 (학습 OOF/앙상블 산출값).
                        ★ 제출용 기본값. val 보고 고르지 않은 정당한 thr.
  THR_MODE = "manual" : 아래 THR_MANUAL 값 사용 (체크/디버깅 전용).
                        ★ 주의: val 보고 thr 고르면 원칙 위반. 분포 확인용으로만.

[그 외]
  - 앙상블 = logit 평균 후 sigmoid (sigmoid 평균 아님).
  - val Rt otitis 양성 슬라이스 raw 확률 진단 출력.
  - 단계형 gate / 후처리 / noise filter / 행기입은 동일.
============================================================
"""
import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

import model as M
from model import get_model, get_temporal_model, get_otitis_model

# ----------------------------- 설정 -----------------------------
BASELINE_CKPT = "best_model.pth"
TEMPORAL_CKPT = "temporal_model.pth"
OTITIS_CKPT = "otitis_model.pth"
OUTPUT = "submission_validation.csv"

USE_NOISE_FILTER = True
MAD_FACTOR = 5.0

GATE_MARGIN = 5
TEMPORAL_PP = {"median_k": 3, "max_gap": 2, "min_run": 2, "keep_largest": False}
OTITIS_PP = {"min_run": 2, "max_gap": 2}

# ======================= threshold 스위치 =======================
#  "auto"   : ckpt threshold 사용 (제출 기본).
#  "manual" : 아래 THR_MANUAL 사용 (체크용). None 인 항목은 ckpt 값으로 자동 fallback.
THR_MODE = "auto"

#  manual 값 (순서: [Rt, Lt]). None 이면 해당 항목만 ckpt 값 사용.
THR_MANUAL = {
    "temporal": [None, None],   # [Rt_temporal, Lt_temporal]
    "otitis":   [None, None],   # [Rt_otitis,   Lt_otitis]
}
# ===============================================================

INFER_TF = M.build_eval_tf()
ROUTE = {("Rt", "temporal area"): ("temporal", 0), ("Lt", "temporal area"): ("temporal", 1),
         ("Rt", "otitis media"): ("otitis", 0), ("Lt", "otitis media"): ("otitis", 1)}


# ----------------------------- 로딩 -----------------------------
def _load_ckpt(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _states(ckpt):
    return ckpt["folds"] if "folds" in ckpt else [ckpt["state_dict"] if "state_dict" in ckpt else ckpt]


def _logits_only(out):
    """OtitisModel(diff_branch) 은 (logits, d) 튜플 반환 → logits 만 사용."""
    return out[0] if isinstance(out, tuple) else out


def _ens(models, fn):
    """fn(model)->logits(B,S,K). ★ logit 평균 후 sigmoid → (S,K) 반환.
    threshold 는 학습 산출값(auto) 또는 manual override 사용."""
    s = None
    with torch.no_grad():
        for m in models:
            lg = fn(m)[0]                      # (S,K) logit (sigmoid 적용 전)
            s = lg if s is None else s + lg
    mean_logit = s / len(models)
    return torch.sigmoid(mean_logit).cpu().numpy()


def _apply_thr_switch(t_thr, o_thr):
    """THR_MODE 에 따라 threshold 확정. manual 이면 None 아닌 항목만 덮어씀."""
    t_thr = list(t_thr); o_thr = list(o_thr)
    if THR_MODE == "manual":
        for i in range(2):
            if THR_MANUAL["temporal"][i] is not None:
                t_thr[i] = float(THR_MANUAL["temporal"][i])
            if THR_MANUAL["otitis"][i] is not None:
                o_thr[i] = float(THR_MANUAL["otitis"][i])
    return t_thr, o_thr


def _tf_full(rt, lt):
    return INFER_TF(rt), INFER_TF(lt)


def _tf_roi(box):
    return lambda rt, lt: (INFER_TF(M.roi_crop(rt, box)), INFER_TF(M.roi_crop(lt, box)))


def load_patient_tensors(dcm_dir, device, exclude, primary_tf, roi_tf=None):
    """환자 폴더 -> idxs, primary(1,S,2,C,H,W), roi or None, meta, pos."""
    exclude = exclude or set()
    idxs, prim, rois, metas, poss = [], [], [], [], []
    for s in range(1, 133):
        if s in exclude:
            continue
        path = os.path.join(dcm_dir, f"{s:04d}.dcm")
        if not os.path.exists(path):
            continue
        image = M.dcm_to_multiwindow_rgb(path)
        rt, lt = M.split_lr(image)
        a, b = primary_tf(rt, lt)
        prim.append(torch.stack([a, b], 0))
        if roi_tf is not None:
            ra, rb = roi_tf(rt, lt)
            rois.append(torch.stack([ra, rb], 0))
        metas.append([s / 132.0]); poss.append(s); idxs.append(s)
    if not idxs:
        return None
    primary = torch.stack(prim, 0).unsqueeze(0).to(device)
    roi = torch.stack(rois, 0).unsqueeze(0).to(device) if roi_tf is not None else None
    meta = torch.tensor(metas, dtype=torch.float).unsqueeze(0).to(device)
    pos = torch.tensor(poss, dtype=torch.long).unsqueeze(0).to(device)
    return idxs, primary, roi, meta, pos


# ----------------------------- 측정 헬퍼 -----------------------------
def _counts(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp)
    tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return tp, fp, fn, 2 * prec * rec / max(prec + rec, 1e-9)


def _load_val_otitis_labels(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    cols = [str(i) for i in range(1, 133)]
    out = {}
    for _, row in df.iterrows():
        if str(row["Image number"]).strip() != "otitis media":
            continue
        out[(int(row["No"]), str(row["R/L"]).strip())] = {
            int(c): int(row[c]) for c in cols if pd.notna(row[c])}
    return out


# ----------------------------- 메인 -----------------------------
def run_inference():
    DATA_ROOT = M.find_data_root()
    TEMPLATE = M.find_file("submission_template.csv")
    VAL_CSV = M.find_file("val_set.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    b_path = M.find_file(BASELINE_CKPT)
    t_path, o_path = M.find_file(TEMPORAL_CKPT), M.find_file(OTITIS_CKPT)

    # ---- ckpt 로딩 우선순위 ----
    b_ck = _load_ckpt(b_path, device) if os.path.exists(b_path) else None
    unified = isinstance(b_ck, dict) and "temporal" in b_ck and "otitis" in b_ck

    if unified:
        mode = "staged"; ckpt_src = f"{BASELINE_CKPT}(unified)"
        t_ck, o_ck = b_ck["temporal"], b_ck["otitis"]
    elif os.path.exists(t_path) and os.path.exists(o_path):
        mode = "staged"; ckpt_src = f"{TEMPORAL_CKPT} + {OTITIS_CKPT}"
        t_ck, o_ck = _load_ckpt(t_path, device), _load_ckpt(o_path, device)
    else:
        mode = "baseline_gate"; ckpt_src = f"{BASELINE_CKPT}(4-output)"

    if mode == "staged":
        t_bb = t_ck.get("backbone", "efficientnet_b0"); o_bb = o_ck.get("backbone", "efficientnet_b0")
        t_models = []
        for sd in _states(t_ck):
            m = get_temporal_model(t_bb, use_diff=t_ck.get("use_diff", True)).to(device); m.load_state_dict(sd); m.eval(); t_models.append(m)
        o_use_roi = o_ck.get("use_roi", False)
        o_roi_box = tuple(o_ck.get("roi_box", M.DEFAULT_OTITIS_ROI))
        o_models = []
        for sd in _states(o_ck):
            m = get_otitis_model(o_bb, use_roi=o_use_roi, use_diff=o_ck.get("use_diff", True)).to(device); m.load_state_dict(sd); m.eval(); o_models.append(m)
        t_thr_ck = t_ck.get("thresholds", [0.5, 0.5]); o_thr_ck = list(o_ck.get("thresholds", [0.5, 0.5]))
        t_pp = dict(t_ck.get("postprocess", TEMPORAL_PP)); gate_margin = t_pp.get("dilation_margin", GATE_MARGIN)
        o_pp = dict(o_ck.get("postprocess", OTITIS_PP))
        primary_tf, roi_tf = _tf_full, (_tf_roi(o_roi_box) if o_use_roi else None)
    else:
        b_bb = b_ck.get("backbone", "efficientnet_b0")
        b_models = []
        for sd in _states(b_ck):
            m = get_model(b_bb, use_diff=b_ck.get("use_diff", True)).to(device); m.load_state_dict(sd); m.eval(); b_models.append(m)
        b_thr = b_ck.get("thresholds", [0.5, 0.5, 0.5, 0.5])
        t_thr_ck = [b_thr[0], b_thr[1]]; o_thr_ck = [b_thr[2], b_thr[3]]
        t_pp = dict(TEMPORAL_PP); gate_margin = GATE_MARGIN; o_pp = dict(OTITIS_PP)
        primary_tf, roi_tf = _tf_roi(M.BASELINE_ROI), None
        t_bb = o_bb = b_bb; o_use_roi = False; o_roi_box = M.BASELINE_ROI

    # ---- threshold 스위치 적용 ----
    t_thr, o_thr = _apply_thr_switch(t_thr_ck, o_thr_ck)

    # ---------------- 설정 로그 ----------------
    print("=" * 64)
    print(f"[v37 추론]  MODE = {mode}  | 앙상블 = logit 평균")
    print(f"  ckpt source         : {ckpt_src}")
    print(f"  device              : {device}")
    print(f"  THR_MODE            : {THR_MODE}")
    if THR_MODE == "manual":
        print(f"    manual temporal   : {THR_MANUAL['temporal']}  (None=ckpt fallback)")
        print(f"    manual otitis     : {THR_MANUAL['otitis']}    (None=ckpt fallback)")
        print(f"    ★ 체크용 모드 — 제출 전 THR_MODE='auto' 로 되돌릴 것")
    if mode == "staged":
        print(f"  TemporalModel ckpt  : (ens {len(t_models)}, {t_bb}, full half)")
        print(f"  OtitisModel   ckpt  : (ens {len(o_models)}, {o_bb}, use_roi={o_use_roi})")
    else:
        print(f"  baseline ckpt       : {BASELINE_CKPT} (4-output, ens {len(b_models)}, {b_bb})")
        print(f"  전처리 ROI(욱여넣기) : {M.BASELINE_ROI}")
    print(f"  HU_WINDOWS          : {M.HU_WINDOWS}")
    print(f"  temporal thr (ckpt) : {[round(float(x),2) for x in t_thr_ck]}  → 적용 {[round(float(x),2) for x in t_thr]} [Rt,Lt]")
    print(f"  otitis   thr (ckpt) : {[round(float(x),2) for x in o_thr_ck]}  → 적용 {[round(float(x),2) for x in o_thr]} [Rt,Lt]")
    print(f"  temporal gate margin: ±{gate_margin}")
    print(f"  temporal smoothing  : median_k={t_pp.get('median_k')} max_gap={t_pp.get('max_gap')} "
          f"min_run={t_pp.get('min_run')} keep_largest={t_pp.get('keep_largest')}")
    print(f"  otitis postprocess  : min_run={o_pp.get('min_run')} max_gap={o_pp.get('max_gap')}")
    print(f"  noise filter        : {USE_NOISE_FILTER} (MAD={MAD_FACTOR})")
    print("=" * 64)

    df = pd.read_csv(TEMPLATE)
    val_labels = _load_val_otitis_labels(VAL_CSV)
    accum = {("before", 0): ([], []), ("before", 1): ([], []),
             ("after", 0): ([], []), ("after", 1): ([], [])}
    diag_all = []
    dcm_not_found = unmatched_rows = 0

    for p_id_raw in tqdm(df["No"].unique(), desc="추론"):
        p_id = str(int(p_id_raw))
        dcm_dir = M.find_patient_dcm_dir(DATA_ROOT, p_id)
        if dcm_dir is None:
            dcm_not_found += 1
            continue

        row_info = {}
        for ridx, row in df[df["No"] == p_id_raw].iterrows():
            r = ROUTE.get((str(row["R/L"]).strip(), str(row["Image number"]).strip()))
            if r is not None:
                row_info[ridx] = r
            else:
                unmatched_rows += 1
        if not row_info:
            continue

        noisy = M.detect_noise_slices(dcm_dir, MAD_FACTOR) if USE_NOISE_FILTER else set()
        loaded = load_patient_tensors(dcm_dir, device, noisy, primary_tf, roi_tf)
        if loaded is None:
            continue
        idxs, primary, roi, meta, pos = loaded
        n = len(idxs)

        # ---------- temporal / otitis 확률 (logit 평균 앙상블) ----------
        if mode == "staged":
            t_probs = _ens(t_models, lambda m: m(primary, meta, pos, pad=None))
            o_probs = _ens(o_models, lambda m: _logits_only(m(primary, meta, pos, pad=None, roi=roi)))
        else:
            probs4 = _ens(b_models, lambda m: m(primary, meta, pos, pad=None))
            t_probs = probs4[:, [0, 1]]
            o_probs = probs4[:, [2, 3]]
     
        # ---------- [진단] val Rt otitis 양성 슬라이스 raw 확률 ----------
        if val_labels is not None:
            lab_rt = val_labels.get((int(p_id_raw), "Rt"))
            if lab_rt:
                pos_probs = [float(o_probs[j, 0]) for j, s in enumerate(idxs)
                             if s in lab_rt and lab_rt[s] == 1]
                if pos_probs:
                    arr = np.array(pos_probs)
                    diag_all.extend(pos_probs)
                    print(f"  [diag] pt{p_id} Rt양성 {len(arr):3d}개  "
                          f"mean={arr.mean():.3f} min={arr.min():.3f} "
                          f"max={arr.max():.3f}  (thr={o_thr[0]:.2f})")

        # ---------- 후처리 (Rt/Lt 독립) ----------
        temporal_pred = np.zeros((n, 2), dtype=int)
        gate = np.zeros((n, 2), dtype=int)
        otitis_pred = np.zeros((n, 2), dtype=int)
        otitis_nogate = np.zeros((n, 2), dtype=int)
        for c in range(2):
            tp_c, gate_c = M.postprocess_temporal(
                t_probs[:, c], threshold=t_thr[c], median_k=t_pp.get("median_k", 3),
                max_gap=t_pp.get("max_gap", 2), min_run=t_pp.get("min_run", 2),
                keep_largest=t_pp.get("keep_largest", False), dilation_margin=gate_margin)
            temporal_pred[:, c], gate[:, c] = tp_c, gate_c
            otitis_pred[:, c] = M.postprocess_otitis(
                o_probs[:, c], threshold=o_thr[c], temporal_gate=gate_c,
                min_run=o_pp.get("min_run", 2), max_gap=o_pp.get("max_gap", 2))
            otitis_nogate[:, c] = M.postprocess_otitis(
                o_probs[:, c], threshold=o_thr[c], temporal_gate=None,
                min_run=o_pp.get("min_run", 2), max_gap=o_pp.get("max_gap", 2))

        # ---------- 행 기입 ----------
        pred_map = {"temporal": temporal_pred, "otitis": otitis_pred}
        idx_set = set(idxs)
        for j, s in enumerate(idxs):
            for ridx, (kind, side) in row_info.items():
                df.at[ridx, str(s)] = int(pred_map[kind][j, side])

        # ---------- noisy slice 복사 ----------
        for s in sorted(noisy):
            if not os.path.exists(os.path.join(dcm_dir, f"{s:04d}.dcm")):
                continue
            nearest = None
            for d in range(1, 132):
                if (s - d) in idx_set:
                    nearest = s - d; break
                if (s + d) in idx_set:
                    nearest = s + d; break
            if nearest is not None:
                for ridx in row_info:
                    df.at[ridx, str(s)] = int(df.at[ridx, str(nearest)])

        # ---------- gate 효과 누적 ----------
        if val_labels is not None:
            for side, rl in ((0, "Rt"), (1, "Lt")):
                lab = val_labels.get((int(p_id_raw), rl))
                if not lab:
                    continue
                for j, s in enumerate(idxs):
                    if s in lab:
                        accum[("before", side)][0].append(lab[s]); accum[("before", side)][1].append(int(otitis_nogate[j, side]))
                        accum[("after", side)][0].append(lab[s]); accum[("after", side)][1].append(int(otitis_pred[j, side]))

    df.to_csv(OUTPUT, index=False)
    print(f"\n추론 완료 → '{OUTPUT}'")
    print(f"  DCM 폴더 탐색 실패 환자 수 : {dcm_not_found}")
    print(f"  ROUTE 매칭 실패 행 수      : {unmatched_rows}")

    # ---------- [진단] 전체 요약 ----------
    if diag_all:
        arr = np.array(diag_all)
        print("\n" + "=" * 64)
        print("   [진단] val Rt otitis 양성 슬라이스 raw 확률 분포")
        print("=" * 64)
        print(f"  총 {len(arr)}개  mean={arr.mean():.3f}  median={np.median(arr):.3f}  "
              f"min={arr.min():.3f}  max={arr.max():.3f}")
        print(f"  적용 thr({o_thr[0]:.2f}) 이상 비율 : {(arr >= o_thr[0]).mean()*100:.1f}%")
        for cut in (0.1, 0.3, 0.5, 0.64):
            print(f"    prob >= {cut:.2f} : {(arr >= cut).mean()*100:5.1f}%")
        print("=" * 64)

    if val_labels is not None and accum[("after", 0)][0]:
        rt_b, rt_a = _counts(*accum[("before", 0)]), _counts(*accum[("after", 0)])
        lt_b, lt_a = _counts(*accum[("before", 1)]), _counts(*accum[("after", 1)])
        print("\n" + "=" * 64)
        print("   temporal gate 적용 전/후 otitis (val_set 기준)")
        print("=" * 64)
        print("[Before gate]")
        print(f"  Rt otitis: TP={rt_b[0]} FP={rt_b[1]} FN={rt_b[2]} F1={rt_b[3]:.4f}")
        print(f"  Lt otitis: TP={lt_b[0]} FP={lt_b[1]} FN={lt_b[2]} F1={lt_b[3]:.4f}")
        print("[After temporal gate]")
        print(f"  Rt otitis: TP={rt_a[0]} FP={rt_a[1]} FN={rt_a[2]} F1={rt_a[3]:.4f}")
        print(f"  Lt otitis: TP={lt_a[0]} FP={lt_a[1]} FN={lt_a[2]} F1={lt_a[3]:.4f}")
        print("[Delta]")
        print(f"  Rt otitis FP change : {rt_a[1]-rt_b[1]:+d}")
        print(f"  Rt otitis FN change : {rt_a[2]-rt_b[2]:+d}")
        print(f"  Rt otitis F1 change : {rt_a[3]-rt_b[3]:+.4f}")
        print(f"  Lt otitis F1 change : {lt_a[3]-lt_b[3]:+.4f}")
        print("=" * 64)
    print(" 정식 채점:  python3 evaluation.py submission_validation.csv")


if __name__ == "__main__":
    run_inference()