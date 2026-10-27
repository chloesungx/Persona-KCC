import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import warnings

from scipy.stats import ConstantInputWarning, chi2_contingency, spearmanr
from sklearn.metrics import mutual_info_score

ANES_VALUE_MAP = {
    "VCF0104": {1: "Male", 2: "Female", 3: "Other"},
    "VCF0105a": {1: "White non-Hispanic", 2: "Black non-Hispanic", 3: "Hispanic", 4: "Other or multiple races, non-Hispanic"},
    "VCF0140a": {1: "8 grades or less (‘grade school’)", 2: "9-12 grades (‘high school’), no diploma/equivalency; less than high school credential (2020)", 3:"12 grades, diploma or equivalency", 4:"12 grades, diploma or equivalency plus non-academic training", 5:"Some college, no degree; junior/community college level degree (AA degree)", 6:"BA level degrees", 7:"Advanced degrees incl. LLB"},
    "VCF0114": {1: "0 to 16 percentile", 2: "17 to 33 percentile", 3: "34 to 67 percentile", 4: "68 to 95 percentile", 5: "96 to 100 percentile"},
    "VCF0112": {1: "Northeast", 2: "North Central", 3: "South", 4: "West"},
    "VCF0116": {1: "Working now", 2: "Temporarily laid off", 4: "Unemployed", 5: "Retired", 6: "Permanently disabled", 7: "Homemaker", 8: "Student"},
    "VCF0128": {1: "Protestant", 2: "Catholic", 3: "Jewish", 4: "Other and none"},
    "VCF0147": {1: "Married", 2: "Never married", 3: "Divorced", 4: "Separated", 5: "Widowed", 7: "Partners; not married"},
    "VCF0148": {0: "Lower class", 1: "Average working", 2: "Working-NA average or upper working", 3: "Upper working", 4: "Average middle", 5: "Middle class-NA average or upper working", 6: "Upper middle", 7: "Upper class"},
    "VCF0146": {1: "Yes, own", 2: "No, not owned"},
    "VCF0301": {1: "Strong Democrat", 2: "Weak Democrat", 3: "Independent-Democrat", 4: "Independent-Independent", 5: "Independent-Republican", 6: "Weak Republican", 7: "Strong Republican"},
    "VCF0305": {1: "Independent", 2: "Leaning Independent", 3: "Weak Partisan", 4: "Strong Partisan"},
    "VCF0605": {0: "나는 정부가 어떤 이익 집단에 의해 운영되는지 대답할 수 없음",
            1: "나는 정부가 몇몇 큰 이익 집단에 의해 운영된다고 생각함",
            2: "나는 정부가 모든 사람들의 이익을 위해 운영된다고 생각함",
            9: "나는 정부가 어떤 이익 집단에 의해 운영되는지 잘 모르겠음"},
    "VCF0606": {0: "나는 정부가 세금으로 낭비하는지 잘 모르겠음",
            1: "나는 정부가 세금을 많이 낭비한다고 생각함",
            2: "나는 정부가 세금을 어느 정도 낭비한다고 생각함",
            3: "나는 정부가 세금을 별로 낭비하지 않는다고 생각함",
            9: "나는 정부가 세금을 낭비하는지 잘 모르겠음"},
    "VCF0803": {1: "Extremely liberal", 2: "Liberal", 3: "Slightly liberal", 4: "Moderate, middle of the road", 5: "Slightly conservative", 6: "Conservative", 7: "Extremely conservative"},
    "VCF0310": {1: "Not much interested", 2: "Somewhat interested", 3: "Very much interested"},
    "VCF0303":{1:"Democrats", 2:"Independents",3:"Republicans"},
    "VCF0154b": {1: "Executive, administrative and managerial", 2: "Professional specialty occupations", 3: "Technicians and related support occupations", 4: "Sales occupation", 5: "Administrative support, including clerical", 6: "Private household", 7: "Protective service", 8: "Service except protective and household", 9: "Farming, forestry and fishing occupations", 10: "Precision production, craft and repair occupations", 11: "Machine operators, assemblers and inspectors", 12: "Transportation and material moving occupations",
13: "Handlers, equipment cleaners, helpers and laborers", 14: "Member of armed forces"},
    "VCF0130": {1: "Every week", 2: "Almost every week", 3:"Once or twice a month", 4:"A few times a year", 5:"Never", 7:"No religious preference"},
    "VCF0156": {1: "Yes", 5: "No"},
    "VCF9279": {1: "Heterosexual or straight", 2:"Bisexual, 3=Homosexual or gay/lesbian", 4:"Something else"}
}

_DEFAULT_THERMOMETER_FIELDS = {
    "VCF0205": "thermometer_jews",
    "VCF0206": "thermometer_blacks",
    "VCF0207": "thermometer_whites",
    "VCF0209": "thermometer_big_business",
    "VCF0210": "thermometer_labor_unions",
    "VCF0211": "thermometer_liberals",
    "VCF0212": "thermometer_conservatives",
    "VCF0214": "thermometer_police",
    "VCF0217": "thermometer_hispanics",
    "VCF0218": "thermometer_democratic_party",
    "VCF0219": "thermometer_middle_class_people",
    "VCF0224": "thermometer_republican_party",
    "VCF0227": "thermometer_asian_americans",
    "VCF0228": "thermometer_congress",
    "VCF0232": "thermometer_gays_lesbians",
    "VCF0233": "thermometer_illegal_aliens",
    "VCF0234": "thermometer_christian_fundamentalists",
    "VCF0253": "thermometer_feminists",
}

_DEFAULT_COLUMN_MAPPING = {
    "VCF0006a": "persona_id",
    "VCF0101": "birth_year",
    "VCF0803": "political_ideology",
    "VCF0301": "party_leaning",
    "VCF0305": "partisanship_strength",
    "VCF0605": "federal_government_view",
    "VCF0606": "government_waste_view",
    "VCF0310": "political_interest",
    # "VCF0133": "region_grew_up",
    # "VCF0148": "social_class",

    "VCF0104": "gender",
    "VCF0105a": "race_ethnicity",
    "VCF0140a":"education_level",
    "VCF0112": "region",
    "VCF0114": "household_income",
    "VCF0116": "occupation_status",
    "VCF0154b": "occupation",
    "VCF0128": "religion",
    "VCF0130": "church_attendance",
    "VCF0146": "home_ownership",
    "VCF0147": "marital_status",
    "VCF0156": "Working_R_Laid_off_Last_6_Months",
    "VCF9279": "sexual_orientation"
    # "VCF0253": "gender_discrimination_perception",
    # "VCF0232": "sexual_minority_discrimination_perception",
    # "VCF0303":"party_affiliation",
}
COLUMN_MAPPING = {
    **_DEFAULT_COLUMN_MAPPING,
    **_DEFAULT_THERMOMETER_FIELDS,
}
THERMOMETER_VCF_CODES = set(_DEFAULT_THERMOMETER_FIELDS)
ANES_MUTABLE_ATTITUDE_FIELDS = [
    "유대인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "흑인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "백인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "대기업에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "노동조합에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "진보주의자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "보수주의자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "경찰에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "히스패닉에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "민주당에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "중산층 사람들에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "공화당에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "아시아계 미국인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "의회에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "게이와 레즈비언에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "불법 이민자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "기독교 근본주의자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "페미니스트에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
]

def _normalize_blank_strings(series: pd.Series) -> pd.Series:
    """Convert whitespace-only string cells to NA before numeric coercion."""
    if series.dtype == object:
        return series.replace(r"^\s*$", np.nan, regex=True)
    return series

def preprocess_anes_2024(csv_path, target_col=None, extra_cols=None): # target_col 인자 추가
    df = pd.read_csv(csv_path, low_memory=False)
    df_2024 = df[df['VCF0004'] == 2024].copy()
    
    target_vcf_keys = list(COLUMN_MAPPING.keys())
    if target_col and target_col not in target_vcf_keys:
        target_vcf_keys.append(target_col)
    for extra_col in extra_cols or []:
        if extra_col and extra_col not in target_vcf_keys:
            target_vcf_keys.append(extra_col)

    for col in target_vcf_keys:
        if col in df_2024.columns:
            df_2024[col] = _normalize_blank_strings(df_2024[col])
            df_2024[col] = pd.to_numeric(df_2024[col], errors='coerce')
            if col in THERMOMETER_VCF_CODES:
                missing_vals = [98, 99]
            else:
                missing_vals = [9, 99, 98]
            
            if col not in ["VCF0101", "VCF0006a"]:
                if col not in THERMOMETER_VCF_CODES and (col not in ANES_VALUE_MAP or 0 not in ANES_VALUE_MAP[col]):
                    missing_vals.append(0)
                if col not in ANES_VALUE_MAP or 8 not in ANES_VALUE_MAP[col]:
                    missing_vals.append(8)
            df_2024[col] = df_2024[col].replace(missing_vals, np.nan)

    for col, mapping in ANES_VALUE_MAP.items():
        if col in df_2024.columns:
            df_2024[col] = _normalize_blank_strings(df_2024[col])
            df_2024[col] = pd.to_numeric(df_2024[col], errors='coerce').astype('Int64').map(mapping)

    if 'VCF0006a' in df_2024.columns:
        df_2024['VCF0006a'] = df_2024['VCF0006a'].fillna(0).astype(int).astype(str)

    # 나이 계산
    # if 'VCF0101' in df_2024.columns:
    #     df_2024['VCF0101'] = 2026 - df_2024['VCF0101']
    df_2024 = df_2024.rename(columns=COLUMN_MAPPING)
            
    final_cols = list(COLUMN_MAPPING.values())
    if target_col and target_col in df_2024.columns:
        final_cols.append(target_col)
    for extra_col in extra_cols or []:
        if extra_col and extra_col in df_2024.columns and extra_col not in final_cols:
            final_cols.append(extra_col)
    
    return df_2024[final_cols]