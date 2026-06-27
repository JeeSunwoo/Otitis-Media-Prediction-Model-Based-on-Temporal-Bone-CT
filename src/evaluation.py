import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict
import sys

# ==========================================================
# 행 정확매칭(No + R/L + Image number) 채점 / macro F1 기준
# ==========================================================
def run_grading(submission_path):
    try:
        ans_df = pd.read_csv('../val_set.csv')
        pred_df = pd.read_csv(submission_path)
    except Exception as e:
        print(f"[Error] 파일 로드 에러: {e}")
        return

    ans_df = ans_df.sort_values(by='No').reset_index(drop=True)

    # 매칭 키 정규화(공백/형 변환으로 인한 매칭 실패 방지)
    for d in (ans_df, pred_df):
        d['R/L'] = d['R/L'].astype(str).str.strip()
        d['Image number'] = d['Image number'].astype(str).str.strip()

    slice_cols = [str(i) for i in range(1, 133)]
    y_true, y_pred = [], []
    cat_true = defaultdict(list)   # 항목별(R/L + Image number) 누적
    cat_pred = defaultdict(list)
    missing_rows = 0

    for idx, row in ans_df.iterrows():
        p_no    = row['No']
        rl      = str(row['R/L']).strip()
        img_num = str(row['Image number']).strip()
        cat_key = f"{rl} / {img_num}"

        # No + R/L + Image number 를 모두 만족하는 예측 행 선택
        pr = pred_df[(pred_df['No'] == p_no) &
                     (pred_df['R/L'] == rl) &
                     (pred_df['Image number'] == img_num)]
        has_row = not pr.empty
        if has_row:
            pr = pr.iloc[0]
        else:
            missing_rows += 1

        for col in slice_cols:
            if pd.notna(row[col]):
                t = int(row[col])
                if not has_row:
                    p = -1                     # 매칭 행 없음 = 오답 처리
                else:
                    try:
                        val = pr[col]
                        if pd.isna(val) or val == "":
                            p = -1
                        else:
                            p = int(float(val))  # 소수점 입력 대비
                    except:
                        p = -1
                y_true.append(t); y_pred.append(p)
                cat_true[cat_key].append(t); cat_pred[cat_key].append(p)

    if len(y_true) == 0:
        print("[Error] 채점할 데이터가 없습니다.")
        return

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='macro', zero_division=0)

    W = 60
    print("=" * W)
    print("  TBCT GRADING RESULT")
    print("=" * W)
    print(f"  Target File : {submission_path}")
    print(f"  Total Slices: {len(y_true)}")
    if missing_rows:
        print(f"  (경고) 매칭 실패 행: {missing_rows}개 → 해당 셀 오답 처리")
    print("-" * W)
    # ===== 메인 점수: macro F1 =====
    print(f">>> macro F1 : {f1:.4f} <<<".center(W))
    print(f"FINAL SCORE: {f1*100:.1f} / 100.0".center(W))
    print(f"(accuracy: {acc*100:.2f}%)".center(W))
    print("-" * W)

    # ===== Detailed Report 1 : 클래스별 =====
    print("\n[ Detailed Report 1 : 클래스별 (Normal / Otitis) ]")
    unique_labels = sorted(set(y_true) | set(y_pred))
    target_names = [('Normal(0)' if l == 0 else 'Otitis(1)' if l == 1 else f'Invalid({l})')
                    for l in unique_labels]
    print(classification_report(y_true, y_pred, labels=unique_labels,
                                target_names=target_names, zero_division=0))

    # ===== Detailed Report 2 : 항목별(R/L + Image number) macro F1 =====
    print("[ Detailed Report 2 : 항목별 (R/L + Image number) macro F1 ]")
    for cat in sorted(cat_true.keys()):
        ct, cp = cat_true[cat], cat_pred[cat]
        cf1  = f1_score(ct, cp, average='macro', zero_division=0)
        cacc = accuracy_score(ct, cp)
        print(f"   {cat:<10s} : F1={cf1:.4f}   acc={cacc*100:6.2f}%   (n={len(ct)})")
    print("=" * W)

    return acc, f1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python3 evaluation.py submission_validation.csv")
    else:
        run_grading(sys.argv[1])
