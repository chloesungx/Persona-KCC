import ast
import os
import json
import time
import pandas as pd
from tqdm import tqdm
import re
import asyncio
import nest_asyncio
nest_asyncio.apply()
import numpy as np
import math
from datetime import datetime, timezone
from dateutil import parser

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.context_extractor import fetch_naver_news_summary
import threading
from src.llm.base import BaseLLM
from openai import OpenAI
from pathlib import Path
from src.kcc.data import search_events

BASE_DIR = Path(__file__).resolve().parents[2] 
env_path = BASE_DIR / ".env"

load_dotenv(env_path)

_BASE = Path(__file__).resolve().parents[2]
PROMPTS_DIR = os.path.join(_BASE, "prompts")
BACKGROUND_MODEL = "gpt-5.4-nano"
current_file = Path(__file__).resolve()
base_dir = current_file.parents[2]
output_dir = str(BASE_DIR / "data" / "output" / "case_D")

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

def get_aggregated_context(event: str, num: int, start_date: str, end_date: str, target_vcf: str = None) -> dict:
    if not target_vcf:
        print("target_vcf가 제공되지 않았습니다. 빈 딕셔너리를 반환합니다.")
        return {}

    file_path = str(BASE_DIR / "data" / "external" / f"data_{target_vcf}.json")

    try:
        if not os.path.exists(file_path):
             print(f"경고: 로컬 파일이 존재하지 않습니다: {file_path}")
             return {}

        with open(file_path, "r", encoding="utf-8") as f:
            local_data = json.load(f)
            
        print(f"로컬 파일 데이터 로드 완료: {file_path} (총 {len(local_data)}개)")
        
        # num 파라미터가 있다면 그만큼 잘라서 반환 (필요시)
        if num and len(local_data) > num:
            # 딕셔너리 슬라이싱 (Python 3.7+ 에서 순서 보장됨)
            return dict(list(local_data.items())[:num])
            
        return local_data

    except Exception as e:
        print(f"로컬 파일 읽기 중 오류 발생 ({file_path}): {e}")
        return {}

def load_prompt(version: str = "v1") -> dict[str, str]:
    path = os.path.join(PROMPTS_DIR, f"{version}.txt")
    with open(path, encoding="utf-8") as f:
        text_content = f.read()
    if "[SYSTEM]" in text_content and "[USER]" in text_content:
        system_part = text_content.split("[SYSTEM]")[1].split("[USER]")[0].strip()
        user_part = text_content.split("[USER]")[1].strip()
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
    선택된 뉴스 제목 리스트(selected_titles)에 대해 
    페르소나 관점에서의 중요도(score)와 생각(thinking)을 분석합니다.
    """
    p_id = persona.get("persona_id", "Unknown")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. 날짜 설정 (시간 감쇠 계산용)
    if val_mode:
        REF_DATE = "2024-12-31T23:59:59+0000"
        HALF_LIFE = 30
    else:
        REF_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z")
        HALF_LIFE = 7

    if not selected_titles:
        return {}

    # 2. LLM 전달용 ID 매핑 (context_0, context_1...)
    title_mapping = {f"context_{i}": title for i, title in enumerate(selected_titles)}
    context_with_ids = "\n".join([f"{tid}: {title}" for tid, title in title_mapping.items()])

    # 3. 프롬프트 구성
    sys_msg = read_context_templates["system"].replace("{profile}", persona['profile']).replace("{context}", context_with_ids)
    usr_msg = read_context_templates["user"].replace("{profile}", persona['profile']).replace("{context}", context_with_ids)

    try:
        # 4. LLM 호출
        response = client.chat.completions.create(
            model=BACKGROUND_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ],
            temperature=0.0,
            max_completion_tokens=5000,
            response_format={"type": "json_object"}
        )
        ans = response.choices[0].message.content
        raw_importance = clean_and_parse_json(ans)
        
        final_importance = {}

        if raw_importance:
            for tid, analysis in raw_importance.items():
                if tid in title_mapping:
                    title = title_mapping[tid]
                    item_meta = data_pool.get(title, {})
                    score = float(analysis.get("score", 0.0))
                    
                    final_importance[title] = {
                        "thinking": analysis.get("thinking", ""),
                        "score": round(score, 4)
                    }
        else:
            print(f"[ID: {p_id}] JSON 분석 실패. LLM 원본 응답:\n{ans}")
            
    except Exception as e:
        print(f"[ID: {p_id}] 중요도 분석 중 예외 발생: {e}")

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

def extract_survey_values(persona, kw_dict, k):
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
    lines = []
    for cat_key in ["query_keyword", "event_keyword"]:
        data = top_context_data.get(cat_key)
        if data:
            lines.append(f"{data['keyword']}")
            for think in data['context']:
                lines.append(f"- {think}")
    return "\n".join(lines) if lines else "관련된 배경지식 없음"


def ask_persona_D(llm: BaseLLM, persona, query, options, event, prompt_templates, kw_dict, k, val_mode=False, history=True):
    p_id = persona.get("persona_id", "Unknown")

    base_profile = persona.get('profile', '')
    previous_choice = persona.get("previous_choice") or "No previous response available"
    first_survey_sentences = extract_survey_values(persona, kw_dict, k) 
    survey_text = "\n".join(first_survey_sentences)
    reference_year="2026"
    current_year = "2026"
    if val_mode ==True:
        reference_year = "2020"
        current_year = "2024"
    
    profile_combined = f"{base_profile}"
    # context_combined = format_top_context(top_context_data)

    profile_combined = f"{base_profile}"
    unique_options = list(dict.fromkeys(options))

    if history:
        fmt = {
            "profile": profile_combined,
            "history": survey_text,
            "pre_choice": previous_choice,
            # "context": context_combined,
            "reference_year": reference_year,
            "current_year": current_year,
            "event": event,
            "query": query,
            "options": "\n".join([f"- {opt}" for opt in unique_options])
        }
    else:
        fmt = {
            "profile": profile_combined,
            "pre_choice": previous_choice,
            # "context": context_combined,
            "reference_year": reference_year,
            "current_year": current_year,
            "event": event,
            "query": query,
            "options": "\n".join([f"- {opt}" for opt in unique_options])
        } 
    
    sys_msg = _fill(prompt_templates["system"], **fmt)
    sys_msg += "\n\nCRITICAL: You MUST output ONLY valid JSON."
    usr_msg = _fill(prompt_templates["user"], **fmt)

    try:
        ans = llm.generate(
            system_prompt=sys_msg,
            user_prompt=usr_msg,
            temperature=0.0,
            max_tokens=4000
        )
        # print('시스템 프롬프트: ',sys_msg)
        print("ans", ans)
        
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

def simulate_event_D(llm: BaseLLM, personas, prompt_templates, read_context_templates, query, event, options, st_bar=None, val_mode=False, target_vcf=None, data=None, history=True):
    start_time = time.time()
    print(f"시뮬레이션 시작 (모드: {'검증' if val_mode else '실시간'})")
    total_personas = len(personas)
    results = {}
    importance_history = {}
    K_VALUE = 10

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if val_mode:
        start_date, end_date = "2024-01-01", "2024-12-31"
    else:
        start_date, end_date = "2026-01-01", "2026-12-31"

    if st_bar: st_bar.progress(0.15, text="외부 정보를 수집하고 있습니다.")
    
    # data_pool = get_aggregated_context(
    #     event=event, 
    #     num=100, 
    #     start_date=start_date, 
    #     end_date=end_date,
    #     target_vcf=target_vcf
    # )

    # if not data_pool:
    #     print("수집된 데이터가 없습니다.")
    #     return {}

    # selected_titles = list(data_pool.keys())[:]
    # print(f"공유 뉴스 풀 생성 완료 (총 {len(selected_titles)}개 제목)")

    if st_bar: st_bar.progress(0.30, text=f"{total_personas}명의 페르소나 응답을 생성합니다.")

    with ThreadPoolExecutor(max_workers=15) as executor:
        def run_pipeline(p):
            # importance_results = evaluate_context_importance(
            #     persona=p, 
            #     selected_titles=selected_titles, 
            #     data_pool=data_pool, 
            #     val_mode=val_mode
            # )
            
            # sorted_items = sorted(
            #     importance_results.items(),
            #     key=lambda x: x[1].get('score', 0),
            #     reverse=True
            # )
            
            # top_thinkings = [item[1].get('thinking', '') for item in sorted_items[:K_VALUE]]
            
            # top_ctx = {
            #     "event_keyword": {
            #         "keyword": event,
            #         "context": top_thinkings
            #     }
            # }
            
            res, _ = ask_persona_D(
                llm, p, query, options, event,
                prompt_templates, {"query_keyword": event}, K_VALUE, val_mode=val_mode, history=history
            )
            return p["persona_id"], res

        futures = [executor.submit(run_pipeline, p) for p in personas]
        
        for i, f in enumerate(as_completed(futures)):
            p_id, res = f.result()
            results[p_id] = {query: res}
            # importance_history[p_id] = imp_res

            # results[p_id] = {
            #     "query": query,
            #     "response": res.get("response"),
            #     "reason": res.get("reason")
            # }

            if st_bar:
                persona_progress = (i + 1) / total_personas
                overall_progress = 0.30 + (0.70 * persona_progress)
                st_bar.progress(
                    min(overall_progress, 1.0), 
                    text=f"시뮬레이션 진행 중: {i + 1}/{total_personas} 완료"
                )
    # output_filename = os.path.join(output_dir, f"{timestamp}_{target_vcf}_B.json")
    # with open(output_filename, 'w', encoding='utf-8') as f:
    #     json.dump(importance_history, f, ensure_ascii=False, indent=4)
    # print(f"\n중요도 분석 결과 저장:\n{output_filename}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("\n" + "="*40)
    print(f"총 소요 시간: {minutes}분 {seconds:.2f}초 ({elapsed_time:.2f}초)")
    print("="*40 + "\n")

    # --- JSON 저장 로직 시작 ---
    filename = f"{timestamp}_{target_vcf if target_vcf else 'results'}_D.json"
    target_dir  = str(BASE_DIR / "data" / "output" / str(target_vcf) / llm.config.model_name / f"history_{history}" / f"{data}_{total_personas}" / "case_D")
    os.makedirs(target_dir, exist_ok=True)

    output_filename = os.path.join(target_dir, filename)
    
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            # results 딕셔너리를 통째로 저장
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n[저장 완료] 페르소나별 응답 결과:\n{output_filename}")
    except Exception as e:
        print(f"파일 저장 중 에러 발생: {e}")
    # --- JSON 저장 로직 끝 ---

    if st_bar: st_bar.progress(1.0, text="시뮬레이션 완료!")
    return results
