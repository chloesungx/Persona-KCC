import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from scipy.stats import entropy
from collections import Counter

LABEL_TO_CODE = {
    "Support tougher environmental regulations": 1,
    "Mostly support tougher environmental regulations": 2,
    "Somewhat support tougher environmental regulations": 3,
    "Moderate view": 4,
    "Somewhat see regulations as too burdensome on business": 5,
    "Mostly see regulations as too burdensome on business": 6,
    "See regulations as too burdensome on business": 7,
}

def calculate_validation_metrics(y_true, y_pred):
    """
    y_true: 실제 정답 리스트 ['Democrats', 'Republicans', ...]
    y_pred: AI 예측 리스트 ['Independents', 'Republicans', ...]
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return None

    # 1. 일대일 대응 지표 (Accuracy, F1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    # 2. 분포 기반 지표 (JS, KL)를 위해 카운트 딕셔너리 생성
    actual_dict = Counter(y_true)
    predicted_dict = Counter(y_pred)
    
    # 두 리스트에 등장하는 모든 범주를 정렬하여 수집
    all_categories = sorted(list(set(actual_dict.keys()) | set(predicted_dict.keys())))
    
    act_counts = np.array([actual_dict.get(cat, 0) for cat in all_categories], dtype=float)
    pre_counts = np.array([predicted_dict.get(cat, 0) for cat in all_categories], dtype=float)
    
    epsilon = 1e-10
    p = (act_counts + epsilon) / (act_counts.sum() + epsilon * len(all_categories))
    q = (pre_counts + epsilon) / (pre_counts.sum() + epsilon * len(all_categories))
    
    kl_div = entropy(p, q)
    m = 0.5 * (p + q)
    js_div = 0.5 * entropy(p, m) + 0.5 * entropy(q, m)
    js_dist = np.sqrt(js_div)

    rmse = None
    try:
        y_true_num = np.array(y_true, dtype=float)
        y_pred_num = np.array(y_pred, dtype=float)
        rmse = np.sqrt(np.mean((y_true_num - y_pred_num) ** 2))
    except (TypeError, ValueError):
        try:
            y_true_num = np.array([LABEL_TO_CODE[label] for label in y_true], dtype=float)
            y_pred_num = np.array([LABEL_TO_CODE[label] for label in y_pred], dtype=float)
            rmse = np.sqrt(np.mean((y_true_num - y_pred_num) ** 2))
        except KeyError:
            rmse = None

    # 3. Confusion Matrix 생성 및 출력
    # labels를 명시하여 행렬의 순서를 all_categories와 동일하게 고정합니다.
    cm = confusion_matrix(y_true, y_pred, labels=all_categories)
    
    # 보기 좋게 pandas DataFrame으로 변환하여 출력
    cm_df = pd.DataFrame(
        cm, 
        index=[f"True:{c}" for c in all_categories], 
        columns=[f"Pred:{c}" for c in all_categories]
    )
    
    print("\n" + "="*40)
    print("           Confusion Matrix")
    print("="*40)
    print(cm_df.to_string())
    print("="*40 + "\n")

    metrics = {
        "Accuracy": round(acc, 4),
        "F1_Score": round(f1, 4),
        "JS_Distance": round(float(js_dist), 4),
        "KL_Divergence": round(float(kl_div), 4),
        "RMSE": round(float(rmse), 4) if rmse is not None else None,
    }

    return metrics, cm_df
