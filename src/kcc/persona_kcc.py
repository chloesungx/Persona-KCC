import os
import json
import ast
import pandas as pd
from dotenv import load_dotenv
from src.conditioning import get_relevant_attributes
from pathlib import Path
import numpy as np       
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] 
env_path = BASE_DIR / ".env"

load_dotenv(env_path)
# DB 설정
SSH_HOST = os.getenv("SSH_HOST")
SSH_PORT = int(os.getenv("SSH_PORT", 4040))
SSH_USER = os.getenv("SSH_USER")
SSH_PASSWORD = os.getenv("SSH_PASSWORD")

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "persona")

# Database access is optional and is initialized by the qualitative workflow only.
tunnel = None
engine = None

current_file = Path(__file__).resolve()
base_dir = current_file.parents[2]

# ── 페르소나 프로필 레이블 ──────────────────────────────
DEMO_LABELS = {
    "birth_year": "출생년도",
    "gender": "성별",
    "marital_status": "혼인 상태",
    "household_income": "가구 총 소득",
    "home_ownership":"주택 보유 여부",
    "education_level": "교육 수준",
    "occupation_status": "직업 상태",
    "occupation": "직업",
    "household_member": "가족수",
    "has_health_insurance": "건강보험 유무",
    "region": "거주 지역",
    "residence_district": "거주 구",
    "political_ideology": "정치적 이데올로기",
    # "party_affiliation": "지지 정당",
    # "party_leaning": "정당 지지 성향",
    "partisanship_strength": "정당 지지 강도",
    "federal_government_view": "연방정부 운영 인식",
    "government_waste_view": "연방정부 세금 낭비 인식",
    # "political_interest": "정치적 관심도",
    "media_trust": "미디어 신뢰도",
    "issue_involvement": "이슈 관여도",
    "political_discussion_frequency": "정치 토론 빈도",
    "religion": "종교",
    "gender_discrimination_perception": "여성 차별 인식",
    "sexual_minority_discrimination_perception": "동성애자 차별 인식",
    "church_attendance": "교회 출석 빈도",
    "Working_R_Laid_off_Last_6_Months": "최근 6개월간 실직 경험",
    "thermometer_jews": "유대인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_blacks": "흑인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_whites": "백인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_big_business": "대기업에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_labor_unions": "노동조합에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_liberals": "진보주의자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_conservatives": "보수주의자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_police": "경찰에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_hispanics": "히스패닉에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_democratic_party": "민주당에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_middle_class_people": "중산층 사람들에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_republican_party": "공화당에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_asian_americans": "아시아계 미국인에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_congress": "의회에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_gays_lesbians": "게이와 레즈비언에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_illegal_aliens": "불법 이민자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_christian_fundamentalists": "기독교 근본주의자에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "thermometer_feminists": "페미니스트에 대한 감정(0:부정, 50: 무관심, 100: 긍정)",
    "network_size": "네트워크 사이즈",
    "happiness_index": "행복지수",
    "life_satisfaction": "삶의 만족도",
    "media_usage": "미디어 이용",
    "sexual_orientation": "성적 지향"
}

THERMOMETER_PROFILE_FIELDS = {
    "thermometer_jews",
    "thermometer_blacks",
    "thermometer_whites",
    "thermometer_big_business",
    "thermometer_labor_unions",
    "thermometer_liberals",
    "thermometer_conservatives",
    "thermometer_police",
    "thermometer_hispanics",
    "thermometer_democratic_party",
    "thermometer_middle_class_people",
    "thermometer_republican_party",
    "thermometer_asian_americans",
    "thermometer_congress",
    "thermometer_gays_lesbians",
    "thermometer_illegal_aliens",
    "thermometer_christian_fundamentalists",
    "thermometer_feminists",
}

EXTERNAL_LABELS = {
    "economic_growth_rate": "경제성장률",
    "gini_coefficient": "지니계수",
    "unemployment_rate": "실업률",
    "crime_rate": "범죄율",
    "news_text": "주요 뉴스 이슈"
}

SURVEY_CSV_PATH = str(base_dir / "data" / "survey" / "persona_profile_survey.csv")
ANES_CSV_PATH = str(base_dir / "data" / "anes_2020_2024_100.csv")

SURVEY_JSON_PATH = str(base_dir / "data" / "survey" / "survey.json")
SURVEY_ANES_JSON_PATH = str(base_dir / "data" / "survey" / "survey_anes_VCF0842.json")

_SURVEY_DF_NORMAL = None
_SURVEY_DF_ANES = None
_SURVEY_MAPPING_NORMAL = None
_SURVEY_MAPPING_ANES = None

def get_survey_data(val_mode=False):
    global _SURVEY_DF_NORMAL, _SURVEY_DF_ANES, _SURVEY_MAPPING_NORMAL, _SURVEY_MAPPING_ANES
 
    if val_mode:
        if _SURVEY_DF_ANES is None:
            if os.path.exists(ANES_CSV_PATH):
                _SURVEY_DF_ANES = pd.read_csv(ANES_CSV_PATH, low_memory=False)
            else:
                _SURVEY_DF_ANES = pd.DataFrame()
 
        if _SURVEY_MAPPING_ANES is None:
            if os.path.exists(SURVEY_ANES_JSON_PATH):
                with open(SURVEY_ANES_JSON_PATH, "r", encoding="utf-8") as f:
                    _SURVEY_MAPPING_ANES = json.load(f)
            else:
                _SURVEY_MAPPING_ANES = {}
 
        return _SURVEY_DF_ANES, _SURVEY_MAPPING_ANES
 
    else:
        if _SURVEY_DF_NORMAL is None:
            if os.path.exists(SURVEY_CSV_PATH):
                _SURVEY_DF_NORMAL = pd.read_csv(SURVEY_CSV_PATH, low_memory=False)
            else:
                _SURVEY_DF_NORMAL = pd.DataFrame()
 
        if _SURVEY_MAPPING_NORMAL is None:
            if os.path.exists(SURVEY_JSON_PATH):
                with open(SURVEY_JSON_PATH, "r", encoding="utf-8") as f:
                    _SURVEY_MAPPING_NORMAL = json.load(f)
            else:
                _SURVEY_MAPPING_NORMAL = {}
 
        return _SURVEY_DF_NORMAL, _SURVEY_MAPPING_NORMAL
    
def extract_survey_context(
    persona_id: str,
    survey_df: pd.DataFrame,
    survey_mapping: dict,
    vcf_keys: list = None,
) -> dict:
    id_col = 'persona_id' if 'persona_id' in survey_df.columns else 'VCF0006a'
    
    # print(vcf_keys)
    persona_row = survey_df[survey_df[id_col].astype(str) == str(persona_id)]
    if persona_row.empty:
        return {}
 
    row_data = persona_row.iloc[0]
    context_by_keyword = {}
 
    for question, mapping_data in survey_mapping.items():
        if vcf_keys is not None and question not in vcf_keys:
            continue
 
        if question not in row_data.index:
            continue
 
        keyword = mapping_data.get("keyword", "기타")
        # print("키워드", keyword)
        descriptions_dict = mapping_data.get("description", {})
        raw_val = row_data[question]
 
        if pd.isna(raw_val) or str(raw_val).strip().lower() == 'nan':
            continue
 
        val = str(raw_val).strip()
        if not val:
            continue
 
        answers = []
        if val.startswith('[') and val.endswith(']'):
            try:
                parsed_list = ast.literal_eval(val)
                if isinstance(parsed_list, list):
                    answers.extend([str(item).strip() for item in parsed_list if str(item).strip()])
                else:
                    answers.append(val)
            except Exception:
                clean_val = val[1:-1]
                answers.extend([x.strip() for x in clean_val.split(',') if x.strip()])
        else:
            answers.append(val)
 
        for ans in answers:
            if ans in descriptions_dict:
                nl_sentence = descriptions_dict[ans]
                if keyword not in context_by_keyword:
                    context_by_keyword[keyword] = []
                context_by_keyword[keyword].append(nl_sentence)
 
    return context_by_keyword

def load_demographic_data(*args, **kwargs) -> pd.DataFrame:
    """Private qualitative database loader (not part of this snapshot)."""
    del args, kwargs
    raise RuntimeError("The private qualitative survey database is not bundled.")


# def load_external_data(engine) -> pd.DataFrame:
#     query = "SELECT * FROM public.external_information ORDER BY timepoint_id;"
#     with engine.connect() as conn:
#         return pd.read_sql(text(query), conn)

def _is_valid(val):
    return pd.notna(val) and str(val).strip() != "" and str(val).lower() != "nan"


def _format_previous_choice(raw_value, choice_map: dict | None = None):
    if not _is_valid(raw_value):
        return None

    if choice_map:
        try:
            code = int(float(raw_value))
            return choice_map.get(code, str(raw_value).strip())
        except Exception:
            pass

    return str(raw_value).strip()

def build_combined_profile_text(row: pd.Series, relevant_attrs: list = None) -> str:
    parts = []
    for col, label in DEMO_LABELS.items():
        if relevant_attrs and col not in relevant_attrs:
            continue
        
        val = row.get(col)
        if _is_valid(val):
            if col == "has_health_insurance":
                val_str = "있음" if val is True or str(val).lower() == 'true' else "없음"
                parts.append(f"{label}: {val_str}")
            elif col in THERMOMETER_PROFILE_FIELDS:
                try:
                    temp = int(float(val))
                    temp_str = "97-100도" if temp == 97 else f"{temp}도"
                except Exception:
                    temp_str = str(val)
                parts.append(f"{label}: {temp_str}")
            else:
                parts.append(f"{label}: {val}")
                
    return ", ".join(parts)

def build_external_context_text(row: pd.Series) -> str:
    parts = []
    for col, label in EXTERNAL_LABELS.items():
        val = row.get(col)
        if _is_valid(val):
            if col == "news_text":
                parts.append(f"[{label}]\n{val}")
            else:
                parts.append(f"{label}: {val}")
    
    return "\n\n".join(parts) if parts else "참조할 특이 외부 정보 없음"

def build_personas(
    df: pd.DataFrame,
    query,
    val_mode=False,
    vcf_keys: list = None,
    prev_vcf: str = None,
    prev_choice_map: dict | None = None,
) -> list[dict]:
    if val_mode==False:
        survey_df, survey_mapping = get_survey_data(val_mode=False)
    else:
        survey_df, survey_mapping = get_survey_data(val_mode=True)
    
    relevant_attrs = None
    if query:
        relevant_attrs = get_relevant_attributes(query)
        if not relevant_attrs:
            relevant_attrs = None

    personas = []
    for _, row in df.iterrows():
        raw_party = row.get("party_leaning")
        valid_party = raw_party if _is_valid(raw_party) else None
        
        persona_id = str(row.get("persona_id", ""))
        previous_choice = _format_previous_choice(row.get(prev_vcf), prev_choice_map) if prev_vcf else None
        
        survey_context_dict = extract_survey_context(
            persona_id=persona_id,
            survey_df=survey_df,
            survey_mapping=survey_mapping,
            vcf_keys=vcf_keys,
        )
        
        personas.append({
            "persona_id": persona_id,
            "profile": build_combined_profile_text(row, relevant_attrs=relevant_attrs),
            "survey_context": survey_context_dict,
            "previous_choice": previous_choice,
            "region": f"{row.get('residence_region', '')} {row.get('residence_district', '')}".strip(),
            "party_leaning": valid_party,
            "source": row.get("source", "original"),
            "last_support": None
        })

        # print("페르소나 프로필:", survey_context_dict)
    return personas
