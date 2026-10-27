import ast
try:
    import streamlit as st
except ModuleNotFoundError:
    st = None
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import re
import asyncio
import random
import nest_asyncio
nest_asyncio.apply()
import numpy as np
import math
from datetime import datetime, timezone
from dateutil import parser

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from src.context_extractor import fetch_naver_news_summary
except ModuleNotFoundError:
    fetch_naver_news_summary = None
import threading
from src.llm.base import BaseLLM
from openai import OpenAI
from pathlib import Path
from src.kcc.data import search_events
try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None

BASE_DIR = Path(__file__).resolve().parents[2] 
env_path = BASE_DIR / ".env"

load_dotenv(env_path)

_BASE = Path(__file__).resolve().parents[2]
PROMPTS_DIR = os.path.join(_BASE, "prompts")
BACKGROUND_MODEL = "gpt-5.4-nano"
current_file = Path(__file__).resolve()
base_dir = current_file.parents[2]
output_dir = str(BASE_DIR / "data" / "output" / "case_A")

def extract_keywords(query, event):
    prompts = load_prompt("keyword_extract_queve")
    
    final_user_msg = prompts["user"].replace("{query}", query).replace("{event}", event)

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=BACKGROUND_MODEL,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": final_user_msg}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        ans = response.choices[0].message.content
        
        ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
        # print(ans_clean)
        return json.loads(ans_clean)
        
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        return {"query_keyword": "정치", "event_keyword": "사회이슈"}

def extract_keywords_val(query, event):
    prompts = load_prompt("keyword_extract_val")
    
    final_user_msg = prompts["user"].replace("{query}", query).replace("{event}", event)

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=BACKGROUND_MODEL,
            messages=[
                {"role": "system", "content": prompts["system"]},
                {"role": "user", "content": final_user_msg}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        ans = response.choices[0].message.content
        
        ans_clean = re.sub(r'^```json\s*|\s*```$', '', ans, flags=re.MULTILINE).strip()
        # print(ans_clean)
        return json.loads(ans_clean)
        
    except Exception as e:
        print(f"키워드 추출 오류: {e}")
        return {"query_keyword": "politic", "event_keyword": "social"}

def load_prompt(version: str = "v1") -> dict[str, str]:
    path = os.path.join(PROMPTS_DIR, f"{version}.txt")
    with open(path, encoding="utf-8") as f:
        text_content = f.read()
    if "[SYSTEM]" in text_content and "[USER]" in text_content:
        system_part = text_content.split("[SYSTEM]")[1].split("[USER]")[0].strip()
        user_part = text_content.split("[USER]")[1].strip()
    elif "<system_instruction>" in text_content and "</system_instruction>" in text_content:
        system_match = re.search(
            r"<system_instruction>(.*?)</system_instruction>",
            text_content,
            flags=re.DOTALL,
        )
        system_part = system_match.group(1).strip() if system_match else text_content.strip()
        user_part = text_content[system_match.end():].strip() if system_match else ""
    else:
        system_part = text_content.strip()
        user_part = ""
    return {"system": system_part, "user": user_part}

class _SafeDict(dict):
    def __missing__(self, key): return "{" + key + "}"

def _fill(template: str, **kwargs):
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    return template

# def get_aggregated_context(event: str, num: int, start_date: str, end_date: str) -> dict:
#     """
#     data.py의 Serper API를 활용하여 주어진 기간 내의 데이터를 긁어옵니다.
#     이벤트 원문을 그대로 검색에 사용하며, 포장 없이 순수 검색 결과만 반환합니다.
#     """
#     if not event:
#         return {}

#     try:
#         serper_results = search_events(
#             event=event, 
#             num=num, 
#             start_date=start_date, 
#             end_date=end_date
#         )
#         return serper_results
        
#     except Exception as e:
#         print(f"Serper API 검색 중 오류 발생 ({event}): {e}")
#         return {}
    
def get_aggregated_context(event: str, num: int, start_date: str, end_date: str, target_vcf: str = None, data:str=None) -> dict:
    """
    Load the local data-pool file for the target event configuration.

    Active behavior:
    - Ignores `event`, `num`, `start_date`, and `end_date` for retrieval logic.
    - Selects one JSON file under `data/data_pool/` based on `target_vcf`
      and optional sample mode (`positive`, `negative`, `neutral`, `balance`).

    Returns:
    - Raw news pool dict keyed by news title.
    - `{}` on missing configuration, missing file, or read failure.
    """
    if not target_vcf:
        print("target_vcf가 제공되지 않았습니다. 빈 딕셔너리를 반환합니다.")
        return {}

    # if os.path.isabs(str(target_vcf)):
    #     file_path = str(target_vcf)
    if data==None:
        file_path = str(BASE_DIR / "data" / "external" / f"data_{target_vcf}.json")
    elif data=='positive':
        file_path = str(BASE_DIR / "data" / "external" / f"data_{target_vcf}_sample_positive.json")
    elif data=='negative':
        file_path = str(BASE_DIR / "data" / "external" / f"data_{target_vcf}_sample_negative.json")
    elif data=='neutral':
        file_path = str(BASE_DIR / "data" / "external" / f"data_{target_vcf}_sample_neutral.json")
    elif data=='balance':
        file_path = str(BASE_DIR / "data" / "external" / f"data_{target_vcf}_sample_balance.json")

    try:
        if not os.path.exists(file_path):
             print(f"경고: 로컬 파일이 존재하지 않습니다: {file_path}")
             return {}

        with open(file_path, "r", encoding="utf-8") as f:
            local_data = json.load(f)
            
        print(f"로컬 파일 데이터 로드 완료: {file_path} (총 {len(local_data)}개)")
        
        # num 파라미터가 있다면 그만큼 잘라서 반환 (필요시)
        # if num and len(local_data) > num:
        #     # 딕셔너리 슬라이싱 (Python 3.7+ 에서 순서 보장됨)
        #     return dict(list(local_data.items())[:num])
            
        return local_data

    except Exception as e:
        print(f"로컬 파일 읽기 중 오류 발생 ({file_path}): {e}")
        return {}
    
def simulate_persona_search(llm, persona, event, data_pool, max_num=10, max_exceeds=10):
    """
    페르소나가 데이터 풀(제목 리스트) 내에서 검색 시뮬레이션을 수행합니다.
    최종 결과로 선택된 뉴스 제목 리스트만 반환합니다.
    """
    title_list = list(data_pool.keys())
    formatted_data = "\n".join([f"- {title}" for title in title_list])

    templates = load_prompt("search_prompt_v.0.1") 
    
    fmt = {
        "persona": persona.get('profile', ''),
        "event": event,
        "data": formatted_data,
        "max_num": max_num,
        "max_exceeds": max_exceeds
    }

    # _fill 함수를 사용하여 프롬프트 완성
    sys_msg = _fill(templates["system"], **fmt)
    usr_msg = _fill(templates["user"], **fmt) if templates.get("user") else "위 지침에 따라 JSON 결과만 출력하세요."

    try:
        # 3. LLM 호출
        ans = llm.generate(
            system_prompt=sys_msg,
            user_prompt=usr_msg,
            temperature=0.0
        )
        
        # 4. JSON 정제 및 파싱
        data = clean_and_parse_json(ans)
        
        if not data or "results" not in data:
            return []

        # 5. 최종 결과에서 news 제목들만 단일 리스트로 추출
        final_kept_titles = []
        for entry in data.get("results", []):
            news_titles = entry.get("news", [])
            for t in news_titles:
                # 중복 제거 및 실제 데이터 풀에 존재하는 제목인지 확인 (선택 사항)
                if t not in final_kept_titles:
                    final_kept_titles.append(t)
        
        # 최대 개수(max_num)만큼 제한하여 반환
        return final_kept_titles[:max_num]

    except Exception as e:
        print(f"Persona Search Simulation Error: {e}")
        return []

from datetime import datetime, timezone


def _render(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template

def coerce_bool(value) -> bool:
    """문자열/숫자 형태로 들어온 bool 유사값을 안정적으로 bool로 변환."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
    return bool(value)

# def simulate_persona_search(llm, persona, event, data_pool, max_num=10, max_exceeds=10):
#     """
#     Stage 1: persona-level retrieval simulation.

#     The function gives the persona profile, target event, and full data pool to
#     an LLM, then asks it to simulate iterative search-and-keep behavior.

#     Returns:
#     - A deduplicated list of kept news titles that actually exist in `data_pool`.
#     - `[]` on prompt/file/JSON failure.
#     """
#     print('[DEBUG] 페르소나 뉴스 탐색 시작')
    
#     if not data_pool:
#         return []

#     # 1. 파일 직접 읽기
#     prompt_path = os.path.join(PROMPTS_DIR, "seach_prompt.txt")
#     if not os.path.exists(prompt_path):
#         print(f"파일을 찾을 수 없습니다: {prompt_path}")
#         return []

#     with open(prompt_path, "r", encoding="utf-8") as f:
#         template = f.read()

#     # 2. 데이터 풀 가공
#     data_json = json.dumps(data_pool, ensure_ascii=False, indent=2)

#     # 3. 렌더링
#     rendered_prompt = _render(
#         template,
#         persona=persona.get('profile', ''),
#         event=event,
#         data=data_json,
#         max_num=max_num,
#         max_exceeds=max_exceeds,
#     )

#     try:
#         ans = llm.generate(
#             system_prompt=rendered_prompt,
#             user_prompt="위의 지침과 데이터 풀을 분석하여 페르소나의 검색 결과(JSON)를 생성하세요.",
#             temperature=0.0
#         )

#         print(f"[DEBUG] LLM 응답 도착")
        
#         # 5. JSON 정제 및 파싱
#         data = clean_and_parse_json(ans)
        
#         if not data or "results" not in data:
#             return []

#         # 6. 결과 리스트 생성
#         final_kept_titles = []
#         for entry in data.get("results", []):
#             news_titles = entry.get("news", [])
#             for t in news_titles:
#                 if t in data_pool and t not in final_kept_titles:
#                     final_kept_titles.append(t)
        
#         print(f"[DEBUG] 최종 선택된 뉴스: {len(final_kept_titles)}건")
#         return final_kept_titles[:max_num]

#     except Exception as e:
#         # 여기서 400 에러가 찍혔던 것입니다.
#         print(f"Persona Search Simulation Error: {e}")
#         return []
    
def clean_and_parse_json(raw_text: str):
    if not raw_text:
        return None
    try:
        text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        text = re.sub(r'^```[a-zA-Z]*\s*|\s*```$', '', text, flags=re.MULTILINE | re.IGNORECASE).strip()
        
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx == -1 or end_idx == -1:
            return None
        json_str = text[start_idx:end_idx+1]

        json_str = json_str.replace('\n', ' ').replace('\r', ' ')        
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        json_str = json_str.replace('" "score"', '"score"').replace('""score"', '"score"')
        json_str = json_str.replace('" "thinking"', '"thinking"').replace('""thinking"', '"thinking"')

        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):

            sanitized = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            try:
                return ast.literal_eval(sanitized)
            except:
                return None
            
    except Exception as e:
        print(f"JSON 정제 중 예상치 못한 오류: {e}")
        return None

def get_time_decay_weight(pub_time_str, reference_date_str=None, half_life_days=7):
    """
    Compute an exponential recency weight for an article timestamp.

    This utility is currently not wired into the active Case A scoring path.
    """
    if not pub_time_str:
            return 1.0
        
    try:
        pub_time = parser.parse(pub_time_str)
        if pub_time.tzinfo is None:
            pub_time = pub_time.replace(tzinfo=timezone.utc)
            
        if reference_date_str:
            ref_date = parser.parse(reference_date_str)
        else:
            ref_date = datetime.now(timezone.utc)
            
        if ref_date.tzinfo is None:
            ref_date = ref_date.replace(tzinfo=timezone.utc)
    
        days_diff = max((ref_date - pub_time).total_seconds() / 86400, 0)
        
        decay_constant = math.log(2) / half_life_days
        decay_weight = math.exp(-decay_constant * days_diff)
        
        return decay_weight
        
    except Exception as e:
        print(f"시간 감쇠 계산 오류: {e}")
        return 1.0

def evaluate_context_importance(persona, selected_titles, data_pool, read_context_templates, val_mode=False):
    """
    Stage 2: persona-level context valuation.

    For each selected news title:
    - fetch its sentence/body from `data_pool` (fallback to description),
    - ask the background model to rate importance one-by-one,
    - collect first-person Korean reactions and normalized scores.

    Returns:
    - `{title: {"thinking": str, "score": float}}`
    - `{}` if there are no selected titles or parsing fails.
    """
    p_id = persona.get("persona_id", "Unknown")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"[DEBUG][A][importance] persona_id={p_id} selected_titles={len(selected_titles)}")
    print(f"[DEBUG][A][importance] template_keys={list(read_context_templates.keys())}")

    # 1. 날짜 설정 (시간 감쇠 계산용)
    if val_mode:
        REF_DATE = "2024-12-31T23:59:59+0000"
        HALF_LIFE = 30
    else:
        REF_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        HALF_LIFE = 7

    if not selected_titles:
        return {}

    final_importance = {}

    for i, title in enumerate(selected_titles):
        item_data = data_pool.get(title, {})
        sentence = item_data.get("sentence", "") or item_data.get("description", "") or title
        context_text = f"Title: {title}\nContent: {sentence}"
        print(f"[DEBUG][A][importance][{p_id}] item={i+1}/{len(selected_titles)} title={title}")
        print(f"[DEBUG][A][importance][{p_id}] sentence_len={len(sentence)}")
        print(f"[DEBUG][A][importance][{p_id}] sentence_preview={sentence[:300]}")

        sys_msg = read_context_templates["system"].replace("{profile}", persona['profile']).replace("{context}", context_text)
        usr_msg = read_context_templates["user"].replace("{profile}", persona['profile']).replace("{context}", context_text)

        try:
            if tiktoken is None:
                raise RuntimeError("tiktoken is not installed")
            try:
                encoding = tiktoken.encoding_for_model(BACKGROUND_MODEL)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")

            sys_tokens = len(encoding.encode(sys_msg))
            usr_tokens = len(encoding.encode(usr_msg))
            total_input_tokens = sys_tokens + usr_tokens

            print(f"\n[Token Info:{p_id}:{i+1}/{len(selected_titles)}] Model: {BACKGROUND_MODEL} (using {encoding.name})")
            print(f"System: {sys_tokens} / User: {usr_tokens} / Total: {total_input_tokens}")

        except Exception as token_err:
            print(f"토큰 계산 중 오류: {token_err}")

        try:
            response = client.chat.completions.create(
                model=BACKGROUND_MODEL,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg}
                ],
                temperature=0.0,
                max_completion_tokens=4000
            )
            ans = response.choices[0].message.content
            print(f"[DEBUG][A][importance][{p_id}] raw_llm_response title={title}: {ans}")
            raw_importance = clean_and_parse_json(ans)
            print(f"[DEBUG][A][importance][{p_id}] parsed_json title={title}: {raw_importance}")

            if isinstance(raw_importance, dict):
                analysis = raw_importance
                nested_keys = [k for k in raw_importance.keys() if str(k).startswith("context_")]
                if nested_keys and isinstance(raw_importance.get(nested_keys[0]), dict):
                    analysis = raw_importance[nested_keys[0]]
                print(f"[DEBUG][A][importance][{p_id}] analysis title={title}: {analysis}")

                score = float(analysis.get("score", 0.0))
                final_importance[title] = {
                    "thinking": analysis.get("thinking", ""),
                    "score": round(score, 4)
                }
                print(f"[DEBUG][A][importance][{p_id}] final_item title={title}: {final_importance[title]}")
            else:
                print(f"[ID: {p_id}] 기사별 중요도 JSON 분석 실패 | title={title}")

        except Exception as e:
            print(f"[ID: {p_id}] 중요도 분석 중 예외 발생 | title={title} | {e}")

    print(f"[분석 완료] ID: {p_id}")
    return final_importance

def build_top_context(importance_data):
    final_result = {}

    for cat_key in ["query_keyword", "event_keyword"]:
        data = importance_data.get(cat_key)
        if not data:
            continue

        actual_keyword = data["keyword"]
        context_map = data["context"]

        sorted_items = sorted(
            context_map.items(), 
            key=lambda x: float(x[1].get('score', 0)), 
            reverse=True
        )

        top_5_thinkings = [
            item[1].get('thinking', '') 
            for item in sorted_items[:5] 
            if item[1].get('thinking')
        ]

        final_result[cat_key] = {
            "keyword": actual_keyword,
            "context": top_5_thinkings
        }

    return final_result

# def extract_survey_values(persona, kw_dict, k):

#     survey_ctx = persona.get("survey_context", {})
#     if not survey_ctx or not isinstance(survey_ctx, dict):
#         return []

#     def collect_sentences(keys):
#         combined = []
#         for key in keys:
#             sentences = survey_ctx.get(key, [])
#             if isinstance(sentences, list):
#                 combined.extend(sentences)
#         return combined

#     try:
#         all_keys = list(survey_ctx.keys())
#         if not all_keys:
#             return []

#         q_kw = kw_dict.get("query_keyword", "")
#         e_kw = kw_dict.get("event_keyword", "")

#         if not q_kw and not e_kw:
#             return collect_sentences(all_keys)

#         client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#         if q_kw and e_kw:
#             texts_to_embed = [q_kw, e_kw] + [str(key) for key in all_keys]
#             response = client.embeddings.create(
#                 model="text-embedding-ada-002",
#                 input=texts_to_embed
#             )

#             embeddings = [np.array(item.embedding) for item in response.data]
#             q_emb = embeddings[0]
#             e_emb = embeddings[1]
#             key_embs = embeddings[2:]

#             def cosine_similarity(a, b):
#                 denom = np.linalg.norm(a) * np.linalg.norm(b)
#                 if denom == 0:
#                     return -1.0
#                 return float(np.dot(a, b) / denom)

#             q_scored = []
#             for key, emb in zip(all_keys, key_embs):
#                 q_scored.append((key, cosine_similarity(q_emb, emb)))
#             q_scored.sort(key=lambda x: x[1], reverse=True)
#             top_k_limit = min(max(int(k), 1), len(q_scored))
#             q_top = [key for key, _ in q_scored[:top_k_limit]]

#             e_scored = []
#             for key, emb in zip(all_keys, key_embs):
#                 e_scored.append((key, cosine_similarity(e_emb, emb)))
#             e_scored.sort(key=lambda x: x[1], reverse=True)
#             e_top = [key for key, _ in e_scored[:top_k_limit]]

#             top_k_list = [key for key in q_top if key in e_top]

#             if not top_k_list:
#                 top_k_list = q_top

#         else:
#             target_kw = q_kw or e_kw
#             texts_to_embed = [target_kw] + [str(key) for key in all_keys]
#             response = client.embeddings.create(
#                 model="text-embedding-ada-002",
#                 input=texts_to_embed
#             )

#             embeddings = [np.array(item.embedding) for item in response.data]
#             target_emb = embeddings[0]
#             key_embs = embeddings[1:]

#             def cosine_similarity(a, b):
#                 denom = np.linalg.norm(a) * np.linalg.norm(b)
#                 if denom == 0:
#                     return -1.0
#                 return float(np.dot(a, b) / denom)

#             scored_keys = []
#             for key, emb in zip(all_keys, key_embs):
#                 sim = cosine_similarity(target_emb, emb)
#                 scored_keys.append((key, sim))

#             scored_keys.sort(key=lambda x: x[1], reverse=True)

#             top_k_limit = min(max(int(k), 1), len(scored_keys))
#             top_k_list = [key for key, _ in scored_keys[:top_k_limit]]

#         # print(top_k_list)

#         return collect_sentences(top_k_list)

#     except Exception as e:
#         print(f"survey_context key top-k 추출 중 오류 발생: {e}")
#         return collect_sentences(list(survey_ctx.keys()))

def extract_survey_values(persona, kw_dict, k):
    """
    Collect all survey-history sentences for the persona.

    Current behavior is intentionally simple:
    - no keyword filtering,
    - no top-k selection,
    - concatenates every sentence in `persona["survey_context"]`.
    """
    survey_ctx = persona.get("survey_context", {})
    if not survey_ctx or not isinstance(survey_ctx, dict):
        return []

    def collect_sentences(keys):
        combined = []
        for key in keys:
            sentences = survey_ctx.get(key, [])
            if isinstance(sentences, list):
                combined.extend(sentences)
        return combined

    try:
        all_keys = list(survey_ctx.keys())
        if not all_keys:
            return []
        return collect_sentences(all_keys)

    except Exception as e:
        print(f"survey_context 추출 중 오류 발생: {e}")
        return collect_sentences(list(survey_ctx.keys()))
    
def format_top_context(top_context_data: dict) -> str:
    """Convert structured top-context data into a prompt-friendly text block."""
    lines = []
    for cat_key in ["query_keyword", "event_keyword"]:
        data = top_context_data.get(cat_key)
        if data:
            lines.append(f"{data['keyword']}")
            for think in data['context']:
                lines.append(f"- {think}")
    return "\n".join(lines) if lines else "관련된 배경지식 없음"


def _section_to_text(value) -> str:
    if not value:
        return ""
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict):
                nested = ", ".join(
                    f"{nested_key}: {nested_value}"
                    for nested_key, nested_value in item.items()
                    if str(nested_value).strip()
                )
                if nested:
                    lines.append(f"{key}: {nested}")
            elif isinstance(item, list):
                joined = "; ".join(str(entry) for entry in item if str(entry).strip())
                if joined:
                    lines.append(f"{key}: {joined}")
            elif str(item).strip():
                lines.append(f"{key}: {item}")
        return "\n".join(lines)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if str(item).strip())
    return str(value).strip()

    
def ask_persona_A(llm: BaseLLM, persona, query, options, event, top_context_data, prompt_templates, kw_dict, k, val_mode=False, history=True):
    """
    Stage 3: final persona answer generation.

    Prompt inputs:
    - persona profile
    - optional survey-history text
    - formatted high-importance context
    - event, survey query, and options

    Returns:
    - `({"response": ..., "reason": ...}, query)`
    - falls back to option-string extraction if JSON parsing fails.
    """
    p_id = persona.get("persona_id", "Unknown")
    history = coerce_bool(history)

    base_profile = persona.get('profile', '')
    demographic = _section_to_text(persona.get("Demographic") or persona.get("demographic"))
    attitude = _section_to_text(persona.get("Attitude") or persona.get("attitude"))
    experience = _section_to_text(persona.get("Experience") or persona.get("experience"))
    previous_choice = persona.get("previous_choice") or "No previous response available"
    first_survey_sentences = extract_survey_values(persona, kw_dict, k) 
    survey_text = "\n".join(first_survey_sentences)
    reference_year="2026"
    current_year = "2026"
    if val_mode ==True:
        reference_year = "2020"
        current_year = "2024"
    
    profile_combined = f"{base_profile}"
    context_combined = format_top_context(top_context_data)
    target_abstract = kw_dict.get("abstract", event) if isinstance(kw_dict, dict) else event
    if not demographic and base_profile:
        demographic = base_profile
    if history and survey_text:
        experience = "\n".join(part for part in [experience, survey_text] if part)

    profile_combined = f"{base_profile}"
    unique_options = list(dict.fromkeys(options))

    fmt = {
        "Demographic": demographic,
        "Attitude": attitude,
        "Experience": experience if history else "",
        "profile": profile_combined,
        "history": survey_text if history else "",
        "pre_choice": previous_choice,
        "context": context_combined,
        "reference_year": reference_year,
        "current_year": current_year,
        "event": event,
        "abstract": target_abstract,
        "query": query,
        "options": "\n".join([f"- {opt}" for opt in unique_options])
    }

    # print(f"[ID: {p_id}] history={history} ({type(history).__name__})")
    # print("입력:", fmt)
    
    sys_msg = _fill(prompt_templates["system"], **fmt)
    sys_msg += "\n\nCRITICAL: You MUST output ONLY valid JSON."
    usr_msg = _fill(prompt_templates["user"], **fmt)

    ans = ""
    try:
        ans = llm.generate(
            system_prompt=sys_msg,
            user_prompt=usr_msg,
            temperature=0.0,
            max_tokens=4000
        )
        # print("ans", ans)
        
        data = clean_and_parse_json(ans)
        
        if data:
            res_val = str(data.get("response", "미응답")).replace("</think>", "").replace("/think", "").strip()
            reason_val = str(data.get("reason", "이유 없음")).replace("</think>", "").replace("/think", "").strip()
            
            return {"response": res_val, "reason": reason_val}, query

        for opt in options:
            if opt in ans:
                print(f"[ID: {p_id}] JSON 실패 방어: 텍스트에서 '{opt}' 강제 추출 완료")
                return {"response": opt, "reason": "JSON 형식 오류로 텍스트에서 직접 추출됨"}, query

        print(f"[ID: {p_id}] 완전히 실패한 응답:\n{ans}")
        return {"response": "미응답", "reason": "데이터 파싱 실패 및 보기 추출 불가"}, query
            
    except Exception as e:
        print(f"[ID: {p_id}] LLM 생성 중 치명적 에러 발생: {e}", ans)
        return {"response": "Error", "reason": str(e)}, query

def simulate_event_A(llm: BaseLLM, personas, prompt_templates, read_context_templates, query, event, options, st_bar=None, val_mode=False, target_vcf=None, data=None, history=True):
    """
    End-to-end orchestrator for Case A.

    Pipeline per persona:
    1. load shared event news pool
    2. simulate persona search
    3. evaluate importance of selected news
    4. build top context from importance-ranked items
    5. generate final persona response
    6. save both responses and importance traces

    Concurrency:
    - personas are processed in parallel with `ThreadPoolExecutor(max_workers=5)`.
    """
    start_time = time.time()
    history = coerce_bool(history)
    print(f"시뮬레이션 시작 (모드: {'검증' if val_mode else '실시간'})")
    print(f"[case A] history={history} ({type(history).__name__})")
    total_personas = len(personas)
    results = {}
    importance_history = {}
    K_VALUE = 10
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if val_mode:
        start_date = "2024-01-01"
        end_date = "2024-12-31"
    else:
        start_date = "2026-01-01"
        end_date = "2026-12-31"

    if st_bar: st_bar.progress(0.10, text="이벤트 관련 데이터를 수집하고 있습니다.")
    
    data_pool = get_aggregated_context(
        event=event, 
        num=100, 
        start_date=start_date, 
        end_date=end_date,
        target_vcf=target_vcf,
        data=data
    )

    if not data_pool:
        print("수집된 데이터가 없어 시뮬레이션을 중단하거나 배경지식 없이 진행합니다.")

    if st_bar: st_bar.progress(0.30, text=f"{total_personas}명의 페르소나가 정보를 탐색하고 응답합니다.")

    with ThreadPoolExecutor(max_workers=5) as executor:
        def run_pipeline(p):
            selected_titles = simulate_persona_search(
                llm, p, event, data_pool, max_num=10, max_exceeds=10
            )

            if not selected_titles and data_pool:
                fallback_titles = list(data_pool.keys())
                selected_titles = random.sample(
                    fallback_titles,
                    k=min(10, len(fallback_titles))
                )
                print(
                    f"[DEBUG][A][pipeline] persona_id={p['persona_id']} "
                    f"fallback_random_selected_titles={selected_titles}"
                )

            print(f"[DEBUG][A][pipeline] persona_id={p['persona_id']} selected_titles={selected_titles}")
            
            importance_results = evaluate_context_importance(
                p, selected_titles, data_pool, read_context_templates, val_mode=val_mode
            )
            print(f"[DEBUG][A][pipeline] persona_id={p['persona_id']} importance_results={importance_results}")
            
            sorted_items = sorted(
                importance_results.items(),
                key=lambda x: x[1].get('score', 0),
                reverse=True
            )
            print(f"[DEBUG][A][pipeline] persona_id={p['persona_id']} sorted_items={sorted_items}")
            top_thinkings = [
                {
                    "thinking": item[1].get('thinking', ''),
                    "score": item[1].get('score', 0)  # score 값이 없을 경우를 대비해 기본값 0 지정
                }
                for item in sorted_items[:K_VALUE]
            ]
            print(f"[DEBUG][A][pipeline] persona_id={p['persona_id']} top_thinkings={top_thinkings}")

            # top_thinkings = [item[1].get('thinking', '') for item in sorted_items[:K_VALUE]]
            
            top_ctx = {
                "event_keyword": {
                    "keyword": event,
                    "context": top_thinkings
                }
            }
            print(f"[DEBUG][A][pipeline] persona_id={p['persona_id']} top_ctx={top_ctx}")
            
            res, _ = ask_persona_A(
                llm, p, query, options, event, top_ctx, 
                prompt_templates, {"query_keyword": event}, K_VALUE, val_mode=val_mode, history=history
            )
            print(f"[DEBUG][A][pipeline] persona_id={p['persona_id']} final_response={res}")
            return p["persona_id"], res, importance_results

        futures = [executor.submit(run_pipeline, p) for p in personas]
        
        for i, f in enumerate(as_completed(futures)):
            p_id, res, imp_res = f.result()
            results[p_id] = {query: res}
            importance_history[p_id] = imp_res
            
            if st_bar:
                persona_progress = (i + 1) / total_personas
                overall_progress = 0.30 + (0.70 * persona_progress)
                st_bar.progress(
                    min(overall_progress, 1.0), 
                    text=f"시뮬레이션 진행 중: {i + 1}/{total_personas} 완료"
                )

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("\n" + "="*40)
    print(f"총 소요 시간: {minutes}분 {seconds:.2f}초 ({elapsed_time:.2f}초)")
    print("="*40 + "\n")

    filename = f"{timestamp}_{target_vcf if target_vcf else 'results'}_results_A.json"
    target_dir  = str(BASE_DIR / "data" / "output" / str(target_vcf) / llm.config.model_name / f"history_{history}" / f"{data}_{total_personas}" / "case_A")
    
    os.makedirs(target_dir, exist_ok=True)

    response_output_filename = os.path.join(target_dir, filename)
    output_filename = os.path.join(target_dir, f"{timestamp}_{target_vcf}_A.json")
    
    try:
        with open(response_output_filename, 'w', encoding='utf-8') as f:
            # results 딕셔너리를 통째로 저장
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n[저장 완료] 페르소나별 응답 결과:\n{response_output_filename}")
    except Exception as e:
        print(f"파일 저장 중 에러 발생: {e}")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(importance_history, f, ensure_ascii=False, indent=4)
    print(f"\n중요도 분석 결과 저장:\n{output_filename}")

    if st_bar: st_bar.progress(1.0, text="시뮬레이션 완료!")
    return results
