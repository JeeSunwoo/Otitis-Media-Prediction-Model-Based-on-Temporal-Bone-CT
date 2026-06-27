"""
model.py  —  v36 통합 모듈 (전처리 + 후처리 + 모델 정의)
============================================================
[3-file 구조]
  model.py     : 공통 CT 전처리/후처리 + backbone builder + 모델들
  train.py     : 단계형 학습 (TemporalModel, OtitisModel)
  inference.py : 단계형 추론 (기존 best_model.pth 욱여넣어 gate 활용 가능)

[설계 의도 — 논문식 pipeline 을 CT 과제로 변환]
  논문(이경 이미지): quality→tympanic seg→laterality→disease.
  CT 변환:
    - segmentation 은 mask GT 가 없으므로 'temporal area slice 구간 detection'으로 대체.
    - laterality 는 CSV Rt/Lt + half split + Lt hflip 으로 대체.
    - disease classification 은 otitis classifier stage 로 변환.
  검증된 동력(반드시 유지):
    (a) DCM HU 멀티윈도우 3채널 입력(PNG 아님)
    (b) 좌우 비대칭 신호를 trunk 입력에 하드코딩(bypass 없음)
  ConvNeXt 는 disease(otitis) backbone 후보로만 도입. EfficientNet-B0 는 baseline.

[모델 구성]
  MyModel       : 4-output [Rt_temp, Lt_temp, Rt_otitis, Lt_otitis] + aux
                  → 기존 best_model.pth(v33/v35) state_dict 와 정확히 호환.
                    재학습 없이 inference 에서 temporal gate 단계형으로 활용.
  TemporalModel : 2-output [Rt_temp, Lt_temp]  (full half, ROI 미사용)
  OtitisModel   : 2-output [Rt_otitis, Lt_otitis] (full half + optional wide ROI)
============================================================
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import pydicom

# ==========================================================
# 전처리 상수 (체크포인트에 저장 → inference 가 그대로 복원)
# ==========================================================
PREPROCESS_VERSION = "hu3win_halfsplit_v36"
IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ch0 bone, ch1 soft tissue, ch2 air/broad — (WL, WW)
HU_WINDOWS = [(700, 2000), (50, 350), (-300, 1400)]
LEFT_HALF_IS_RT = False # axial view 기준 이미지 오른쪽 절반 = 환자 Rt

# otitis 보조 ROI 후보 (hflip 정렬 후 반쪽 기준 비율). wide 부터 실험.
ROI_CANDIDATES = {
    "R1": (0.10, 0.05, 1.00, 0.95),
    "R2": (0.15, 0.10, 1.00, 0.95),
    "R3": (0.20, 0.10, 1.00, 0.95),
    "NARROW": (0.28, 0.20, 0.98, 0.85),   # 기존 best_model.pth 학습 ROI (비교/호환용)
}
DEFAULT_OTITIS_ROI = ROI_CANDIDATES["R1"]
# best_model.pth(v33/v35)가 학습된 ROI. 그 모델을 추론에 쓸 때 반드시 이 박스 사용.
BASELINE_ROI = ROI_CANDIDATES["NARROW"]


# ==========================================================
# PART A.  CT 전처리 / 후처리 공통 함수
#   train.py / inference.py 가 반드시 같은 코드를 import → 전처리 불일치 차단
# ==========================================================
def read_dcm_hu(dcm_path):
    """RescaleSlope/Intercept 적용한 HU(float32)."""
    d = pydicom.dcmread(dcm_path)
    return d.pixel_array.astype(np.float32) * float(d.get("RescaleSlope", 1)) \
        + float(d.get("RescaleIntercept", 0))


def apply_window(hu, wl, ww):
    """HU 를 window level/width 로 0~1 clip(float32)."""
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def dcm_to_multiwindow_rgb(dcm_path):
    """HU 복원 → HU_WINDOWS 3개를 0~255 uint8 3채널 PIL Image."""
    hu = read_dcm_hu(dcm_path)
    chans = [(apply_window(hu, wl, ww) * 255).astype(np.uint8) for wl, ww in HU_WINDOWS]
    return Image.fromarray(np.stack(chans, axis=-1), mode="RGB")


def split_lr(img, left_half_is_rt=LEFT_HALF_IS_RT):
    """슬라이스 1장 -> (Rt_half, Lt_half). Lt 는 hflip 으로 Rt 와 방향 정렬.
    ★ ROI 는 자르지 않음 (roi_crop 에서 별도 처리)."""
    w, h = img.size
    half = w // 2
    left = img.crop((0, 0, half, h)); right = img.crop((half, 0, w, h))
    rt, lt = (left, right) if left_half_is_rt else (right, left)
    return rt, TF.hflip(lt)


def roi_crop(half_img, roi_box):
    """반쪽 이미지에서 normalized box (left,top,right,bottom) 비율로 crop."""
    w, h = half_img.size
    l, t, r, b = roi_box
    return half_img.crop((int(round(l * w)), int(round(t * h)),
                          int(round(r * w)), int(round(b * h))))


def build_eval_tf():
    return transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                               transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def _dcm_gray(dcm_path):
    hu = read_dcm_hu(dcm_path)
    return apply_window(hu, 40, 1500)


def _mad_threshold(values, factor):
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return factor * 1.4826 * mad


def detect_noise_slices(dcm_dir, mad_factor=5.0):
    """'올라갔다 바로 내려오는' 노이즈 슬라이스 번호 set (MAD 기반, v35 유지)."""
    slices = {}
    for i in range(1, 133):
        path = os.path.join(dcm_dir, f"{i:04d}.dcm")
        if os.path.exists(path):
            slices[i] = _dcm_gray(path)
    if len(slices) < 5:
        return set()
    diffs = {}
    for num in sorted(slices.keys()):
        nbs = [np.mean(np.abs(slices[num] - slices[num + off]))
               for off in (-1, 1) if num + off in slices]
        if nbs:
            diffs[num] = float(np.mean(nbs))
    nums = sorted(diffs.keys())
    if len(nums) < 3:
        return set()
    deltas = {nums[i]: diffs[nums[i]] - diffs[nums[i - 1]] for i in range(1, len(nums))}
    pos_thr = _mad_threshold(list(deltas.values()), mad_factor)
    noisy = set()
    dnums = sorted(deltas.keys())
    for i in range(len(dnums) - 1):
        cur, nxt = dnums[i], dnums[i + 1]
        if deltas[cur] > pos_thr and deltas[nxt] < -pos_thr:
            noisy.add(cur)
    diff_vals = np.array([diffs[n] for n in nums])
    hi = np.median(diff_vals) + _mad_threshold(diff_vals, mad_factor)
    if diffs[nums[0]] > hi:
        noisy.add(nums[0])
    if diffs[nums[-1]] > hi:
        noisy.add(nums[-1])
    return noisy


# ---------------- 시퀀스 후처리 (1D numpy, 슬라이스 순서, Rt/Lt 독립) ----------------
def _runs(arr):
    arr = np.asarray(arr).astype(int)
    runs, i, n = [], 0, len(arr)
    while i < n:
        if arr[i] == 1:
            j = i
            while j < n and arr[j] == 1:
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def binary_median_smooth(arr, k=3):
    arr = np.asarray(arr).astype(int)
    if k <= 1 or len(arr) == 0:
        return arr.copy()
    r = k // 2; out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i - r), min(len(arr), i + r + 1)
        out[i] = 1 if arr[lo:hi].sum() * 2 > (hi - lo) else 0
    return out


def fill_small_gaps(arr, max_gap=2):
    arr = np.asarray(arr).astype(int); out = arr.copy(); n = len(arr); i = 0
    while i < n:
        if out[i] == 0:
            j = i
            while j < n and out[j] == 0:
                j += 1
            if (i - 1 >= 0 and out[i - 1] == 1) and (j < n and out[j] == 1) and (j - i) <= max_gap:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


def remove_short_runs(arr, min_len=2):
    out = np.asarray(arr).astype(int).copy()
    for s, e in _runs(out):
        if (e - s) < min_len:
            out[s:e] = 0
    return out


def keep_largest_component(arr):
    arr = np.asarray(arr).astype(int); runs = _runs(arr)
    if not runs:
        return arr.copy()
    out = np.zeros_like(arr); s, e = max(runs, key=lambda r: r[1] - r[0]); out[s:e] = 1
    return out


def dilate_1d_mask(arr, margin=5):
    arr = np.asarray(arr).astype(int)
    if margin <= 0:
        return arr.copy()
    out = arr.copy()
    for s, e in _runs(arr):
        out[max(0, s - margin):min(len(arr), e + margin)] = 1
    return out


def postprocess_temporal(prob, threshold, median_k=3, max_gap=2, min_run=2,
                         keep_largest=False, dilation_margin=5):
    """temporal 확률 시퀀스 → (temporal_pred, gate).
    temporal_pred : temporal area row 기입용 (dilation 미적용)
    gate          : otitis 후보 zone (temporal_pred 를 ±dilation_margin 확장)."""
    binm = (np.asarray(prob, dtype=np.float64) >= threshold).astype(int)
    binm = binary_median_smooth(binm, median_k)
    binm = fill_small_gaps(binm, max_gap)
    binm = remove_short_runs(binm, min_run)
    if keep_largest:
        binm = keep_largest_component(binm)
    return binm, dilate_1d_mask(binm, dilation_margin)


def postprocess_otitis(prob, threshold, temporal_gate=None, min_run=2, max_gap=2):
    """otitis 확률 시퀀스 → threshold → gate AND → gap fill → short-run 제거."""
    binm = (np.asarray(prob, dtype=np.float64) >= threshold).astype(int)
    if temporal_gate is not None:
        binm = binm * np.asarray(temporal_gate).astype(int)
    binm = fill_small_gaps(binm, max_gap)
    return remove_short_runs(binm, min_run)


# ---------------- 경로 탐색 헬퍼 ----------------
def find_data_root():
    for c in ("./data", "../data", "../../data"):
        if os.path.isdir(c):
            return c
    return "../data"


def find_file(name):
    for d in (".", "..", "../.."):
        c = os.path.join(d, name)
        if os.path.exists(c):
            return c
    return name


def find_patient_dcm_dir(data_root, p_id, subdirs=("", "val", "test", "train")):
    for sub in subdirs:
        cand = os.path.join(data_root, p_id, "DCM") if sub == "" \
            else os.path.join(data_root, sub, p_id, "DCM")
        if os.path.isdir(cand):
            return cand
    return None


# ==========================================================
# PART B.  backbone builder (eff_b0 / eff_b3 / convnext_tiny)
# ==========================================================
def _ctor_with_weights(ctor, enum):
    for w in (enum, "IMAGENET1K_V1", None):
        try:
            return ctor(weights=w)
        except Exception:
            continue
    return ctor(weights=None)


def build_backbone(name):
    """backbone 이름 -> (feature_extractor, feat_dim). classifier 제거 (GAP 이후 벡터)."""
    if name == "efficientnet_b0":
        net = _ctor_with_weights(models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = net.classifier[1].in_features; net.classifier = nn.Identity()
    elif name == "efficientnet_b3":
        net = _ctor_with_weights(models.efficientnet_b3, models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        feat_dim = net.classifier[1].in_features; net.classifier = nn.Identity()
    elif name == "convnext_tiny":
        net = _ctor_with_weights(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        feat_dim = net.classifier[2].in_features
        net.classifier = nn.Sequential(net.classifier[0], net.classifier[1])  # norm+flatten
    else:
        raise ValueError(f"unknown backbone: {name}")
    return net, feat_dim


SPATIAL_PROJ_DIM = 256   # pre-GAP 1280ch 공간맵 -> 1x1 conv 투영 차원 (메모리 통제)


class SpatialBackbone(nn.Module):
    """efficientnet_b0 의 pre-GAP 공간맵을 꺼내는 backbone.
    net.features(x) -> (N,1280,7,7) -> 공유 1x1 conv -> (N, proj_dim, 7,7).
    GAP 를 적용하지 않아 국소 병변/위치정보 보존 (OtitisModel 의 좌우 공간비교용).
    gradient checkpointing 으로 backbone unfreeze 후 VRAM 통제 (과거 OOM 이력).
    ★ efficientnet_b0 전용 (다른 backbone 은 이번 범위 아님)."""
    def __init__(self, backbone_name="efficientnet_b0", proj_dim=SPATIAL_PROJ_DIM):
        super().__init__()
        if backbone_name != "efficientnet_b0":
            raise ValueError(f"SpatialBackbone 은 efficientnet_b0 전용 (got {backbone_name})")
        net = _ctor_with_weights(models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = net.features            # (N,1280,7,7)
        self.feat_ch = net.classifier[1].in_features
        self.proj = nn.Conv2d(self.feat_ch, proj_dim, kernel_size=1)
        self.proj_dim = proj_dim
        self.use_checkpoint = True

    def forward(self, x):
        if (self.use_checkpoint and self.training
                and any(p.requires_grad for p in self.features.parameters())):
            feat = cp.checkpoint(self.features, x, use_reentrant=False)
        else:
            feat = self.features(x)
        return self.proj(feat)                  # (N, proj_dim, 7,7)


# ==========================================================
# PART C.  MyModel — 4-output (기존 best_model.pth 호환)
#   logits 순서 [Rt_temp(0), Lt_temp(1), Rt_otit(2), Lt_otit(3)], aux=좌우통합 otitis
# ==========================================================
D_MODEL, N_HEADS, N_LAYERS, META_DIM, MAX_SLICES = 128, 4, 2, 1, 140
SIDE_TO_LABEL = {0: (0, 2), 1: (1, 3)}   # side0=Rt -> (Rt_temp,Rt_otit), side1=Lt


class MyModel(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, meta_dim=META_DIM, use_diff=True):
        super().__init__()
        self.use_diff = use_diff
        self.backbone, feat_dim = build_backbone(backbone_name)
        self.feat_dim = feat_dim
        self.trunk = nn.Sequential(
            nn.Linear(feat_dim * 2 + meta_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, d_model), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.3))
        self.pos_embed = nn.Embedding(MAX_SLICES, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=True, activation="gelu")
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.temporal_head = nn.Linear(d_model, 1)
        self.otitis_head = nn.Linear(d_model, 1)
        self.aux_otitis = nn.Linear(d_model * 2, 1)

    def _encode_side(self, feat_self, feat_other, meta, pos, pad, B, S):
        diff = (feat_self - feat_other) if self.use_diff else torch.zeros_like(feat_self)
        enc = self.trunk(torch.cat([feat_self, diff, meta], dim=1)).view(B, S, -1)
        enc = enc + self.pos_embed(pos)
        return self.seq_encoder(enc, src_key_padding_mask=pad)

    def forward(self, imgs, meta, pos, pad=None):
        B, S, two, C, H, W = imgs.shape
        feat = self.backbone(imgs.view(B * S * 2, C, H, W)).view(B, S, 2, self.feat_dim)
        fR = feat[:, :, 0].reshape(B * S, self.feat_dim)
        fL = feat[:, :, 1].reshape(B * S, self.feat_dim)
        meta_f = meta.reshape(B * S, -1)
        encR = self._encode_side(fR, fL, meta_f, pos, pad, B, S)
        encL = self._encode_side(fL, fR, meta_f, pos, pad, B, S)
        logits = torch.zeros(B, S, 4, device=imgs.device)
        for side, enc in ((0, encR), (1, encL)):
            t_idx, o_idx = SIDE_TO_LABEL[side]
            logits[:, :, t_idx] = self.temporal_head(enc).squeeze(-1)
            logits[:, :, o_idx] = self.otitis_head(enc).squeeze(-1)
        aux = self.aux_otitis(torch.cat([encR, encL], dim=-1)).squeeze(-1)
        return logits, aux


def get_model(backbone_name="efficientnet_b0", use_diff=True):
    return MyModel(backbone_name=backbone_name, use_diff=use_diff)


# ==========================================================
# PART D.  단계형 모델 (TemporalModel / OtitisModel)
# ==========================================================
class _SeqCore(nn.Module):
    """좌우 diff trunk + position embed + Transformer (Temporal/Otitis 공유)."""
    def __init__(self, side_dim, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, meta_dim=META_DIM, use_diff=True):
        super().__init__()
        self.use_diff = use_diff
        self.trunk = nn.Sequential(
            nn.Linear(side_dim * 2 + meta_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, d_model), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.3))
        self.pos_embed = nn.Embedding(MAX_SLICES, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.2, batch_first=True, activation="gelu")
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, feat_self, feat_other, meta, pos, pad, B, S):
        diff = (feat_self - feat_other) if self.use_diff else torch.zeros_like(feat_self)
        enc = self.trunk(torch.cat([feat_self, diff, meta], dim=1)).view(B, S, -1)
        enc = enc + self.pos_embed(pos)
        return self.seq_encoder(enc, src_key_padding_mask=pad)


class TemporalModel(nn.Module):
    """full half only. 출력 (B,S,2) [Rt_temp, Lt_temp]."""
    def __init__(self, backbone_name="efficientnet_b0", use_diff=True):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone_name)
        self.feat_dim = feat_dim
        self.core = _SeqCore(feat_dim, use_diff=use_diff)
        self.head = nn.Linear(D_MODEL, 1)

    def forward(self, imgs, meta, pos, pad=None):
        B, S, two, C, H, W = imgs.shape
        feat = self.backbone(imgs.view(B * S * 2, C, H, W)).view(B, S, 2, self.feat_dim)
        fR = feat[:, :, 0].reshape(B * S, self.feat_dim)
        fL = feat[:, :, 1].reshape(B * S, self.feat_dim)
        meta_f = meta.reshape(B * S, -1)
        encR = self.core(fR, fL, meta_f, pos, pad, B, S)
        encL = self.core(fL, fR, meta_f, pos, pad, B, S)
        logits = torch.zeros(B, S, 2, device=imgs.device)
        logits[:, :, 0] = self.head(encR).squeeze(-1)
        logits[:, :, 1] = self.head(encL).squeeze(-1)
        return logits


class _SpatialSideEncoder(nn.Module):
    """좌우 공간맵을 '공간에서' 비교해 per-side feature 벡터를 만든다 (좌우 공유 가중치).
    입력: self_map, other_map (N, C, h, w)  — split_lr 의 Lt hflip 으로 위치 대응 정렬됨.
    concat[self_map, self_map - other_map] -> conv -> 공간 pooling -> (N, out_dim).
    self_map(절대 증거) + 좌우 공간차(상대 증거)를 함께 보고 per-side 표현 생성.
    encR = f(R, L), encL = f(L, R) 로 동일 함수 공유 → 대칭."""
    def __init__(self, in_ch, mid_ch=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 2, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.out_dim = mid_ch

    def forward(self, self_map, other_map):
        x = torch.cat([self_map, self_map - other_map], dim=1)
        x = self.conv(x)
        return F.adaptive_avg_pool2d(x, 1).flatten(1)      # (N, out_dim)


class _SeqTransformer(nn.Module):
    """per-side feature 시퀀스 -> position embed -> Transformer (좌우 공유).
    좌우 비교는 이미 _SpatialSideEncoder 공간단계에서 끝났으므로 여기서 벡터 diff 는 하지 않음."""
    def __init__(self, in_dim, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, meta_dim=META_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim + meta_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, d_model), nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(0.4))
        self.pos_embed = nn.Embedding(MAX_SLICES, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=0.3, batch_first=True, activation="gelu")
        self.seq_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, feat, meta, pos, pad, B, S):
        enc = self.trunk(torch.cat([feat, meta], dim=1)).view(B, S, -1)
        enc = enc + self.pos_embed(pos)
        return self.seq_encoder(enc, src_key_padding_mask=pad)


class _DiffBranch(nn.Module):
    """antisymmetric diff branch (좌우 공간맵 공유).
    f(X,Y) = small_conv(X - Y) -> GAP -> scalar.  (GAP 전에 conv 로 국소 좌우차 처리)
    d = f(R,L) - f(L,R)  → 구조적으로 d(R,L) = -d(L,R) (side hardcoding 없는 antisymmetry).
    tanh 로 (-1,1) bound. 슬라이스별 스칼라."""
    def __init__(self, in_ch, mid_ch=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU())
        self.fc = nn.Linear(mid_ch, 1)

    def _f(self, x, y):
        z = self.conv(x - y)
        z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        return self.fc(z).squeeze(-1)                      # (N,)

    def forward(self, R_map, L_map):
        return torch.tanh(self._f(R_map, L_map) - self._f(L_map, R_map))


class OtitisModel(nn.Module):
    """pre-GAP 공간 좌우비교 기반 otitis classifier. 출력 (B,S,2) [Rt_otit, Lt_otit] 과 d(B,S).

    판정 자체가 좌우 공간비교 위에서 일어남 (사후 보정 아님):
      backbone(SpatialBackbone) -> 좌우 공간맵 (N,proj,7,7)
      main : _SpatialSideEncoder 로 self vs (self-other) 공간차 -> per-side feature
             -> 공유 _SeqTransformer -> per-side logit
      diff : _DiffBranch antisymmetric d (B,S), tanh-bound
      결합 : Rt_final = Rt_logit + alpha*d ,  Lt_final = Lt_logit - alpha*d  (alpha 학습)
    use_roi 인자는 단계형 호환을 위해 받지만 공간경로에서는 사용하지 않음(현행 OTITIS_USE_ROI=False).
    """
    def __init__(self, backbone_name="efficientnet_b0", use_roi=False, use_diff=True):
        super().__init__()
        self.backbone = SpatialBackbone(backbone_name)       # 좌우 공유 (Siamese)
        self.proj_dim = self.backbone.proj_dim
        self.use_roi = use_roi                               # 공간경로 미사용 (ckpt 기록용)
        self.use_diff = use_diff                             # diff branch 가산 결합 토글(ablation)
        self.side_spatial = _SpatialSideEncoder(self.proj_dim)
        self.core = _SeqTransformer(self.side_spatial.out_dim)
        self.head = nn.Linear(D_MODEL, 1)
        self.diff_branch = _DiffBranch(self.proj_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))           # 0 근처에서 시작

    def forward(self, full, meta, pos, pad=None, roi=None):
        B, S = full.shape[:2]
        C, H, W = full.shape[-3:]
        maps = self.backbone(full.view(B * S * 2, C, H, W))  # (B*S*2, proj, h, w)
        pj, fh, fw = maps.shape[-3:]
        maps = maps.view(B, S, 2, pj, fh, fw)
        R_map = maps[:, :, 0].reshape(B * S, pj, fh, fw)
        L_map = maps[:, :, 1].reshape(B * S, pj, fh, fw)
        meta_f = meta.reshape(B * S, -1)

        vR = self.side_spatial(R_map, L_map)                 # (B*S, side_dim)
        vL = self.side_spatial(L_map, R_map)
        encR = self.core(vR, meta_f, pos, pad, B, S)
        encL = self.core(vL, meta_f, pos, pad, B, S)
        rt_logit = self.head(encR).squeeze(-1)               # (B,S)
        lt_logit = self.head(encL).squeeze(-1)

        d = self.diff_branch(R_map, L_map).view(B, S)        # (B,S) ∈ (-1,1)
        final = torch.zeros(B, S, 2, device=full.device)
        if self.use_diff:
            a = self.alpha.clamp(min=0.2)        # ★ 하한 0.2 (diff 안 꺼지게)
            final[:, :, 0] = rt_logit + a * d
            final[:, :, 1] = lt_logit - a * d
        else:
            final[:, :, 0] = rt_logit
            final[:, :, 1] = lt_logit
        return final, d


def get_temporal_model(backbone_name="efficientnet_b0", use_diff=True):
    return TemporalModel(backbone_name=backbone_name, use_diff=use_diff)


def get_otitis_model(backbone_name="efficientnet_b0", use_roi=False, use_diff=True):
    return OtitisModel(backbone_name=backbone_name, use_roi=use_roi, use_diff=use_diff)
