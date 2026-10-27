import ast
from pathlib import Path
import os
import warnings
import transformers

# 1. 허깅페이스의 불필요한 정보성/경고성 로그를 끔 (에러 났을 때만 출력)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
transformers.logging.set_verbosity_error()

# 2. 파이썬 기본 경고(FutureWarning, UserWarning 등) 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="transformers")

CACHE_DIR = str(Path(__file__).resolve().parents[2] / "checkpoints" / ".cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR

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
import tiktoken

from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.context_extractor import fetch_naver_news_summary
import threading
from src.llm.base import BaseLLM
from openai import OpenAI
from pathlib import Path
from src.kcc.data import search_events
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
import torch
from typing import List, Dict, Any, Optional
from collections import defaultdict

# ---------- 모델 로더 (전역 캐시) ----------
TWHIN_MODEL_NAME = "Twitter/twhin-bert-large"
_twhin_model: Optional[AutoModel] = None
_twhin_tokenizer: Optional[AutoTokenizer] = None

BASE_DIR = Path(__file__).resolve().parents[2] 
env_path = BASE_DIR / ".env"

load_dotenv(env_path)

_BASE = Path(__file__).resolve().parents[2]
PROMPTS_DIR = os.path.join(_BASE, "prompts")
PROXY_PATH = str(BASE_DIR / "data" / "network" / "proxy_network20.json")

current_file = Path(__file__).resolve()
base_dir = current_file.parents[2]
output_dir = str(BASE_DIR / "data" / "output" / "case_C")
BACKGROUND_MODEL = "gpt-5.4-nano"
NEWS_EXPLORATION_TEMPERATURE = float(os.getenv("NEWS_EXPLORATION_TEMPERATURE", "0.0"))
LOCAL_TWHIN_PATH = str(BASE_DIR / "checkpoints")
_client = None
DEBUG_ACTION_PROMPT = os.getenv("DEBUG_ACTION_PROMPT", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "y",
)


def _truncate_for_log(text: str, max_chars: int = 6000) -> str:
    text = str(text or "")
    if max_chars and len(text) > int(max_chars):
        return text[: int(max_chars)] + "..."
    return text


def build_agent_system_prompt(
    persona_profile: str,
    name: str | None = None,
    history=None,
) -> str:
    name_string = f"Your name is {name}." if name else ""
    description_lines: list[str] = []
    if persona_profile:
        description_lines.append(f"Your have profile: {persona_profile}.")
    if history:
        try:
            history_text = (
                json.dumps(history, ensure_ascii=False)
                if isinstance(history, (dict, list))
                else str(history)
            )
        except Exception:
            history_text = str(history)
        history_text = history_text.strip()
        if history_text:
            max_chars = 1200
            if len(history_text) > max_chars:
                history_text = history_text[:max_chars] + "..."
            description_lines.append(f"Here is your history: {history_text}.")

    description_string = "\n".join(description_lines).strip()
    description = f"{name_string}\n{description_string}".strip()

    system_content = f"""
# OBJECTIVE
You're a Twitter user, and I'll present you with some tweets. After you see the tweets, choose some actions from the following functions.

# SELF-DESCRIPTION
Your actions should be consistent with your self-description and personality.
{description}

# RESPONSE METHOD
Please perform actions by tool calling.
        """
    return system_content


def build_agent_user_prompt(env_prompt: str) -> str:
    return (
        "After observing the platform environment, choose the single action that best matches your natural behavior. "
        "It is okay to do nothing when nothing stands out. "
        f"Here is your social media environment: {env_prompt}"
    )


def build_env_prompt_from_news(
    news_data_pool: dict,
    max_items: int | None = None,
    *,
    json_indent: int | None = 4,
    max_title_chars: int = 240,
) -> str:
    groups_env = "No groups."

    titles = list(news_data_pool.keys())
    if max_items is not None:
        titles = titles[:max_items]

    min_create_posts = min(5, len(titles))

    posts = []
    for i, title in enumerate(titles, start=1):
        title_str = str(title or "")
        if max_title_chars and len(title_str) > int(max_title_chars):
            title_str = title_str[: int(max_title_chars)] + "..."

        posts.append({
            "post_id": i,
            "user_id": "news_pool",
            "content": title_str,
            "created_at": "",
            "num_likes": 0,
        })

    posts_env = (
        "After refreshing, you see some posts "
        + json.dumps(posts, ensure_ascii=False, indent=json_indent)
    )

    env_prompt = (
        f"{groups_env}\n"
        f"{posts_env}\n"
        "Consider ALL posts above and decide how you would naturally react to each one.\n"
        "This is a multi-post feed, not a single-post decision.\n"
        f"You MUST choose `create_post` for at least {min_create_posts} posts in the entire feed.\n"
        f"If fewer than {min_create_posts} posts feel strongly relevant, choose the {min_create_posts} most relevant, noticeable, or personally meaningful posts anyway.\n"
        "A `create_post` can be a short casual reaction, question, concern, agreement, disagreement, or takeaway.\n"
        "A `create_post` does not need to be a strong or expert-level opinion.\n"
        "Do not be overly passive. Do not choose `do_nothing` for almost everything.\n"
        "Use `like_post` or `dislike_post` only when your reaction is simple and you would not say anything publicly.\n"
        "Use `do_nothing` only for posts that are clearly irrelevant or do not trigger any meaningful reaction.\n"
        f"Before finalizing your answer, check that at least {min_create_posts} actions are `create_post`.\n"
    )

    return env_prompt 


def build_env_prompt_for_one_news(
    *,
    title: str,
    article: str,
    post_id: int = 1,
    max_article_chars: int = 800,
) -> str:
    groups_env = "No groups."
    if isinstance(article, dict):
        link = article.get("link", "")
        desc = article.get("description", "")
        article = f"{desc}\n{link}".strip()
    else:
        article = str(article or "").strip()
    if max_article_chars and len(article) > int(max_article_chars):
        article = article[: int(max_article_chars)] + "..."

    posts = [{
        "post_id": int(post_id),
        "user_id": "news_pool",
        "content": str(title),
        "article": article,
        "created_at": "",
        "num_likes": 0,
    }]

    posts_env = (
        "After refreshing, you see some posts "
        + json.dumps(posts, ensure_ascii=False, indent=4)
    )

    # Note: we keep the same outer structure as the original environment prompt,
    # but restrict the decision to THIS single post.
    env_prompt = (
        f"{groups_env}\n"
        f"{posts_env}\n"
        "You are currently viewing one post in your feed.\n"
        "Decide how you would naturally react to this post.\n"
        "Possible reactions include creating your own post, liking it, disliking it, or doing nothing.\n"
        "Use `do_nothing` for clearly irrelevant posts, but do not be overly passive. In a feed with many posts, it is natural to create a few short posts about the items that feel most relevant to you. \n"
        "Use `like_post` for a mild positive reaction or agreement when you do not feel like saying anything publicly.\n"
        "Use `create_post` when the post is personally relevant enough that you would make a brief public reaction, quote-post, comment, question, or takeaway. A `create_post` does not need to be a strong opinion; it can be a short casual reaction.\n"
        "Use `dislike_post` for a mild negative reaction, disagreement, or discomfort when you do not feel like saying anything publicly.\n"
        "Use `do_nothing` when the post does not matter to you or does not trigger any meaningful reaction.\n"
        "Choose based primarily on how strongly you feel compelled to express yourself publicly, not just on whether your reaction is positive or negative.\n"
        "Choose the action that best matches your natural behavior."
    )
    return env_prompt


SOCIAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "create_post",
            "description": (
                "Create a new post when the viewed content feels worth sharing or commenting on publicly. "
                "Write a very short original post (1 sentence). "
                "Set `content` to what you would actually post (can include the title)."
            ),
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "like_post",
            "parameters": {
                "type": "object",
                "properties": {"post_id": {"type": "integer"}},
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dislike_post",
            "parameters": {
                "type": "object",
                "properties": {"post_id": {"type": "integer"}},
                "required": ["post_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

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
            temperature=0.7,
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

def get_aggregated_context(event: str, num: int, start_date: str, end_date: str, target_vcf: str = None, data:str=None) -> dict:
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

def _call_gpt_json(system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict:
    """OpenAI JSON 모드 호출 후 dict로 반환."""
    client = _get_client()
    response = client.chat.completions.create(
        model=BACKGROUND_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    ans = response.choices[0].message.content
    try:
        return json.loads(ans)
    except json.JSONDecodeError:
        cleaned = ans.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)

def _render(template: str, **kwargs) -> str:
    for key, value in kwargs.items():
        template = template.replace("{" + key + "}", str(value))
    return template

_render = _fill

def run_news_exploration(
    personas: List[dict],
    news_data_pool: dict,
    max_workers: int = 5,
    temperature: float | None = None,
) -> Dict[str, dict]:
    """
    Decide actions for a batch of news items, per persona.

    Output:
        {persona_id: {"created_post": [title, ...]}}

    This version enforces at least 5 create_post selections per persona
    when the feed contains 5 or more posts.
    """
    titles = list(news_data_pool.keys())
    min_create_posts = min(5, len(titles))

    created_by_persona: dict[str, list[str]] = defaultdict(list)
    action_counts_by_persona: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    def _decide_batch(persona: dict) -> tuple[str, list[str], dict[str, int]]:
        persona_id = str(persona.get("persona_id", "")).strip()
        if not persona_id:
            return "", [], {"invalid_persona": 1}

        env_prompt = build_env_prompt_from_news(
            news_data_pool,
            max_items=len(titles),
            json_indent=None,
        )

        persona_profile = persona.get("profile", "") or ""
        history = persona.get("survey_context", "")

        system_msg = (
            "You are simulating an ordinary Twitter/X user, not a silent survey respondent.\n"
            "You will be shown a feed of posts. Each post has a numeric post_id and a content string.\n\n"

            "For EACH post_id, choose exactly ONE action from: create_post, like_post, dislike_post, do_nothing.\n\n"

            "CRITICAL ACTION CONSTRAINT:\n"
            f"- You MUST choose `create_post` for at least {min_create_posts} posts in this feed.\n"
            f"- Prefer choosing `create_post` for about {min_create_posts} to 8 posts, unless many posts are highly relevant.\n"
            f"- If fewer than {min_create_posts} posts feel strongly relevant, still choose the {min_create_posts} most relevant, noticeable, or personally meaningful posts for `create_post`.\n"
            "- Do NOT choose `do_nothing` too often just because your opinion is mild.\n"
            "- A mild concern, agreement, disagreement, question, emotional reaction, or personal takeaway is enough for `create_post`.\n"
            f"- Before finalizing your answer, check that at least {min_create_posts} actions are `create_post`.\n\n"

            "Action guidelines:\n"
            "- Use `create_post` when the post is personally relevant enough that you would make a brief public reaction, quote-post, comment, question, concern, agreement, disagreement, or takeaway.\n"
            "- A `create_post` does NOT require a strong, expert-level, or highly original opinion. It can be a casual one-sentence reaction.\n"
            "- Use `like_post` when you mildly agree with or appreciate the post, but do not have anything to add publicly.\n"
            "- Use `dislike_post` when you mildly disagree with or feel uncomfortable about the post, but do not want to comment publicly.\n"
            "- Use `do_nothing` only when the post feels clearly irrelevant, uninteresting, or not worth reacting to.\n\n"

            "When writing `create_post` content:\n"
            "- Write only ONE short sentence.\n"
            "- Write in the persona's own voice.\n"
            "- Do not sound like a news summary.\n"
            "- Do not mention that you are an AI, persona, simulation, or language model.\n\n"

            "Return ONLY valid JSON, with this exact schema:\n"
            "{\n"
            '  "actions": [\n'
            '    {"post_id": 1, "action": "do_nothing"},\n'
            '    {"post_id": 2, "action": "create_post", "content": "..." }\n'
            "  ]\n"
            "}\n\n"

            "JSON requirements:\n"
            "- Do not omit any post_id.\n"
            "- Every post_id must appear exactly once.\n"
            "- Every action must be one of: create_post, like_post, dislike_post, do_nothing.\n"
            "- Every `create_post` action must include a non-empty `content` field.\n"
            f"- The final JSON must contain at least {min_create_posts} items whose action is `create_post`.\n\n"

            f"Persona profile:\n{persona_profile}\n"
            f"History (optional):\n{history}\n"
        )

        user_msg = env_prompt

        if DEBUG_ACTION_PROMPT:
            try:
                print("\n" + "=" * 80)
                print(f"[DEBUG_ACTION_PROMPT] persona_id={persona_id} (BATCH)")
                print("\n--- system ---\n" + _truncate_for_log(system_msg))
                print("\n--- user ---\n" + _truncate_for_log(user_msg))
                print("=" * 80 + "\n")
            except Exception as e:
                print(f"[DEBUG_ACTION_PROMPT] print failed: {e}")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        temp = NEWS_EXPLORATION_TEMPERATURE if temperature is None else float(temperature)

        resp = client.chat.completions.create(
            model=BACKGROUND_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=temp,
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or "{}"
        parsed = clean_and_parse_json(raw) or {}
        actions = parsed.get("actions", [])

        if not isinstance(actions, list):
            return persona_id, [], {"parse_failed": 1}

        created_titles: list[str] = []
        created_post_ids: set[int] = set()
        action_by_post_id: dict[int, str] = {}
        counts: dict[str, int] = defaultdict(int)

        # 1. Parse model actions.
        for item in actions:
            if not isinstance(item, dict):
                counts["invalid_action_item"] += 1
                continue

            try:
                post_id = int(item.get("post_id"))
            except Exception:
                counts["invalid_post_id"] += 1
                continue

            if not (1 <= post_id <= len(titles)):
                counts["invalid_post_id"] += 1
                continue

            # If duplicated, ignore later duplicates.
            if post_id in action_by_post_id:
                counts["duplicate_post_id"] += 1
                continue

            action = str(item.get("action", "")).strip()

            if action not in ("create_post", "like_post", "dislike_post", "do_nothing"):
                counts["invalid_action_name"] += 1
                continue

            action_by_post_id[post_id] = action
            counts[action] += 1

            if action == "create_post":
                created_post_ids.add(post_id)
                created_titles.append(titles[post_id - 1])

        # 2. Treat omitted post_ids as do_nothing.
        for post_id in range(1, len(titles) + 1):
            if post_id not in action_by_post_id:
                action_by_post_id[post_id] = "do_nothing"
                counts["do_nothing"] += 1

        # 3. Hard fallback:
        # If the model selected fewer than min_create_posts, force-fill from the most reactive candidates.
        # Priority: like_post -> dislike_post -> do_nothing.
        if len(created_post_ids) < min_create_posts:
            needed = min_create_posts - len(created_post_ids)

            candidate_ids: list[int] = []
            for priority_action in ("like_post", "dislike_post", "do_nothing"):
                for post_id in range(1, len(titles) + 1):
                    if post_id in created_post_ids:
                        continue
                    if action_by_post_id.get(post_id) == priority_action:
                        candidate_ids.append(post_id)

            for post_id in candidate_ids[:needed]:
                old_action = action_by_post_id.get(post_id, "do_nothing")

                if counts.get(old_action, 0) > 0:
                    counts[old_action] -= 1

                action_by_post_id[post_id] = "create_post"
                counts["create_post"] += 1
                counts["forced_min_create_post"] += 1

                created_post_ids.add(post_id)
                created_titles.append(titles[post_id - 1])

            print(
                f"[MIN_CREATE_POST 보정] persona_id={persona_id} "
                f"create_post를 최소 {min_create_posts}개로 보정했습니다. "
                f"forced={counts.get('forced_min_create_post', 0)}"
            )

        # 4. De-dup while preserving order.
        seen: set[str] = set()
        ordered: list[str] = []
        for title in created_titles:
            if title not in seen:
                ordered.append(title)
                seen.add(title)

        return persona_id, ordered, dict(counts)

    max_workers = max(int(max_workers), 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_decide_batch, persona) for persona in personas]

        for f in tqdm(as_completed(futures), total=len(futures), desc="[1단계] 뉴스 탐색(병렬)"):
            try:
                persona_id, created_titles, counts = f.result()

                if not persona_id:
                    continue

                for k, v in (counts or {}).items():
                    action_counts_by_persona[persona_id][k] += int(v)

                if created_titles:
                    created_by_persona[persona_id].extend(created_titles)

            except Exception as e:
                print(f"[1단계 오류] {e}")

    results: dict[str, dict] = {}

    for persona in personas:
        persona_id = str(persona.get("persona_id", "")).strip()
        if not persona_id:
            continue

        seen: set[str] = set()
        ordered: list[str] = []

        for title in created_by_persona.get(persona_id, []):
            if title not in seen:
                ordered.append(title)
                seen.add(title)

        results[persona_id] = {"created_post": ordered}

        counts = action_counts_by_persona.get(persona_id, {})
        if counts:
            print(
                f"[1단계 액션 요약] persona_id={persona_id} "
                f"create_post={counts.get('create_post', 0)}, "
                f"like_post={counts.get('like_post', 0)}, "
                f"dislike_post={counts.get('dislike_post', 0)}, "
                f"do_nothing={counts.get('do_nothing', 0)}, "
                f"forced_min_create_post={counts.get('forced_min_create_post', 0)}, "
                f"parse_failed={counts.get('parse_failed', 0)}"
            )

    return results


def generate_opinions(
    personas: List[dict],
    exploration_results: Dict[str, dict],
    news_data_pool: dict,
    max_workers: int = 5,
) -> Dict[str, Dict[str, dict]]:
    print("2단계 진입")
    print(f"[DEBUG] 전달받은 페르소나 총 개수: {len(personas)}")
    
    if not personas:
        print("[DEBUG] 페르소나 리스트가 비어있습니다. 함수를 종료합니다.")
        return {}
    prompt_path = os.path.join(PROMPTS_DIR, "opinion_generation_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()

    def _format_history(history, max_chars: int = 1200) -> str:
        if not history:
            return ""
        try:
            history_text = (
                json.dumps(history, ensure_ascii=False)
                if isinstance(history, (dict, list))
                else str(history)
            )
        except Exception:
            history_text = str(history)
        history_text = history_text.strip()
        if not history_text:
            return ""
        if max_chars and len(history_text) > int(max_chars):
            history_text = history_text[: int(max_chars)] + "..."
        return history_text

    def _call_gpt_json_with_client(
        client: OpenAI,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> dict:
        response = client.chat.completions.create(
            model=BACKGROUND_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        ans = response.choices[0].message.content or "{}"
        try:
            return json.loads(ans)
        except json.JSONDecodeError:
            cleaned = ans.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(cleaned)

    def _opinion_one(persona: dict) -> tuple[str, Dict[str, dict]]:
        persona_id = str(persona.get("persona_id", "")).strip()
        if not persona_id:
            return "", {}

        titles = exploration_results.get(persona_id, {}).get("created_post", []) or []
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        post_map: Dict[str, dict] = {}

        for idx, title in enumerate(titles, start=1):
            raw_article = news_data_pool.get(title, "")
            if isinstance(raw_article, dict):
                article = str(raw_article.get("description", "") or "").strip()
            else:
                article = str(raw_article or "").strip()
            history_text = _format_history(persona.get("survey_context", None))
            persona_text = (persona.get("profile", "") or "").strip()
            if history_text:
                persona_text = (persona_text + "\n\n[HISTORY]\n" + history_text).strip()
            rendered = _render(
                template,
                persona=persona_text,
                title=title,
                article=article,
            )
            try:
                data = _call_gpt_json_with_client(
                    client,
                    system_prompt=rendered,
                    user_prompt="해당 뉴스에 대한 의견을 JSON으로만 출력하세요.",
                    temperature=0.0,
                )
                content = (data.get("content") or "").strip()
            except Exception as e:
                print(f"[2단계 오류] {persona_id}/{title}: {e}")
                content = ""

            post_id = f"post_{persona_id}_{idx:03d}"
            post_map[post_id] = {"title": title, "content": content}

        return persona_id, post_map

    results: Dict[str, Dict[str, dict]] = {}
    max_workers = max(int(max_workers), 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_opinion_one, p) for p in personas]
        for f in tqdm(as_completed(futures), total=len(futures), desc="[2단계] 의견 생성(병렬)"):
            try:
                persona_id, post_map = f.result()
                if not persona_id:
                    continue
                results[persona_id] = post_map
            except Exception as e:
                print(f"[2단계 오류] {e}")

    # Ensure all personas have a result entry.
    for persona in personas:
        persona_id = str(persona.get("persona_id", "")).strip()
        if persona_id and persona_id not in results:
            results[persona_id] = {}

    # print("2단계 결과 생성: ", results)
    return results

def collect_network_contents(
    persona_id: str,
    proxy_data: Dict[str, dict],
    opinions_results: Dict[str, Dict[str, dict]],
) -> List[Dict[str, Any]]:
    persona_info = proxy_data.get(str(persona_id), {})
    network = [str(n) for n in persona_info.get("network", [])]
    collected = []
    for neighbor_id in network:
        neighbor_posts = opinions_results.get(str(neighbor_id), {})
        for post_id, post in neighbor_posts.items():
            if not post.get("content"):
                continue
            collected.append({
                "post_id": post_id,
                "title": post.get("title", ""),
                "content": post["content"],
                "author": str(neighbor_id),
            })
    return collected


def _get_twhin():
    global _twhin_model, _twhin_tokenizer
    
    if _twhin_model is None or _twhin_tokenizer is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        _twhin_tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_TWHIN_PATH, 
            local_files_only=True
        )
        
        _twhin_model = AutoModel.from_pretrained(
            LOCAL_TWHIN_PATH, 
            local_files_only=True
        ).to(device)
        
        _twhin_model.eval()
        print("[System] 로컬 TwHIN-BERT 모델 오프라인 로딩 완료!")
        
    return _twhin_model, _twhin_tokenizer


# ---------- 임베딩 ----------
@torch.no_grad()
def _process_batch(model, tokenizer, batch_texts: List[str]) -> torch.Tensor:
    device = next(model.parameters()).device
    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    return outputs.pooler_output  # (B, hidden)


def generate_post_vector(texts: List[str], batch_size: int = 100) -> torch.Tensor:
    model, tokenizer = _get_twhin()
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = [t if (t and isinstance(t, str)) else "empty" for t in texts[i:i + batch_size]]
        all_outputs.append(_process_batch(model, tokenizer, batch))
    return torch.cat(all_outputs, dim=0).cpu()


# ---------- 피드 생성 ----------
def generate_feed(
    persona_id: str,
    proxy_data: Dict[str, dict],
    network_contents: List[Dict[str, Any]],
    top_k: Optional[int] = None,
    batch_size: int = 100,
) -> List[Dict[str, Any]]:
    bio = (proxy_data.get(persona_id, {}).get("bio") or "").strip()
    if not bio or not network_contents:
        return []

    contents_text = [c["content"] for c in network_contents]
    corpus = [bio] + contents_text
    vectors = generate_post_vector(corpus, batch_size=batch_size)

    bio_vec = vectors[:1].numpy()
    content_vecs = vectors[1:].numpy()
    sims = sk_cosine_similarity(bio_vec, content_vecs)[0]

    ranked = sorted(
        [{**c, "score": float(s)} for c, s in zip(network_contents, sims)],
        key=lambda x: x["score"],
        reverse=True,
    )
    print("해당 페르소나의 피드:", persona_id, ranked)
    return ranked[:top_k] if top_k else ranked

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

def evaluate_context_importance(
    persona: dict,
    selected_contents: list[str],
    val_mode: bool = False,
) -> dict:
    """
    피드에 올라온 상위 N개의 컨텐츠(이웃의 의견 등)에 대해 
    페르소나 관점에서의 중요도(score)와 생각(thinking)을 분석합니다.
    """
    p_id = persona.get("persona_id", "Unknown")
    v11_templates = load_prompt("v13_read_context")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not selected_contents:
        return {}

    # 1. LLM 전달용 ID 매핑 (context_0, context_1...)
    content_mapping = {f"context_{i}": c for i, c in enumerate(selected_contents)}
    context_with_ids = "\n".join([f"{cid}: {c}" for cid, c in content_mapping.items()])

    # 2. 프롬프트 구성
    sys_msg = v11_templates["system"].replace("{profile}", persona.get("profile", "")).replace("{context}", context_with_ids)
    usr_msg = v11_templates["user"].replace("{profile}", persona.get("profile", "")).replace("{context}", context_with_ids)

    try:
        # 3. LLM 호출 (JSON Object 포맷 강제)
        response = client.chat.completions.create(
            model=BACKGROUND_MODEL,
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ],
            temperature=0.0,
            max_completion_tokens=6000,
            response_format={"type": "json_object"}
        )
        ans = response.choices[0].message.content
        raw_importance = clean_and_parse_json(ans)
        
        final_importance = {}

        # 4. 결과 매핑 및 정리
        if raw_importance:
            for cid, analysis in raw_importance.items():
                if cid in content_mapping:
                    content_key = content_mapping[cid]
                    score = float(analysis.get("score", 0.0))
                    
                    final_importance[content_key] = {
                        "thinking": analysis.get("thinking", ""),
                        "score": round(score, 4)
                    }
        else:
            print(f"[ID: {p_id}] JSON 분석 실패. LLM 원본 응답:\n{ans}")
            
    except Exception as e:
        print(f"[ID: {p_id}] 중요도 분석 중 예외 발생: {e}")

    # print(f"[분석 완료] ID: {p_id}")
    return final_importance

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


    
def ask_persona_C(llm: BaseLLM, persona, query, options, event, top_context_data, prompt_templates, kw_dict, k, val_mode=False, history=True):
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
    context_combined = format_top_context(top_context_data)

    profile_combined = f"{base_profile}"
    unique_options = list(dict.fromkeys(options))

    if history:
        fmt = {
            "profile": profile_combined,
            "history": survey_text,
            "pre_choice": previous_choice,
            "context": context_combined,
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
            "context": context_combined,
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

def simulate_event_C(
    llm: BaseLLM,
    personas: List[dict],
    prompt_templates: dict,
    read_context_templates,
    query: str,
    event: str,
    options: list,
    st_bar=None,
    val_mode: bool = False,
    feed_top_n: int = 10,
    llm_max_workers: int = 10,
    target_vcf=None,
    data=None,
    history=True
):
    start_time = time.time()
    print(f"시뮬레이션 시작 (모드: {'검증' if val_mode else '실시간'})")
    selected_personas = personas
    total_personas = len(selected_personas)
    K_VALUE = 10
    results = {}
    importance_history = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if val_mode:
        start_date, end_date = "2024-01-01", "2024-12-31"
    else:
        start_date, end_date = "2026-01-01", "2026-12-31"

    if st_bar:
        st_bar.progress(0.05, text="외부 정보를 수집하고 있습니다.")
    data_pool = get_aggregated_context(
        event=event, num=100, start_date=start_date, end_date=end_date, target_vcf=target_vcf, data=data
    )
    if not data_pool:
        print("수집된 데이터가 없습니다.")
        return {}
    print(f"공유 뉴스 풀 생성 완료 (총 {len(data_pool)}개 제목)")

    with open(PROXY_PATH, "r", encoding="utf-8") as f:
        proxy_data = json.load(f)

    # Expand opinion generation set to include 1-hop network neighbors so that
    # small selected samples can still form a feed with enough neighbor content.
    personas_by_id: dict[str, dict] = {
        str(p.get("persona_id")): p for p in selected_personas if p.get("persona_id") is not None
    }
    expanded_ids: set[str] = set(personas_by_id.keys())
    for pid in list(expanded_ids):
        neighbors = proxy_data.get(str(pid), {}).get("network", []) or []
        for nid in neighbors:
            expanded_ids.add(str(nid))

    opinion_personas: list[dict] = []
    for pid in expanded_ids:
        existing = personas_by_id.get(pid)
        if existing is not None:
            opinion_personas.append(existing)
            continue
        # Fallback persona info for neighbors not in `selected_personas`:
        # use proxy bio as the "profile" so prompts still have some persona signal.
        fallback_profile = (proxy_data.get(pid, {}).get("bio") or "").strip()
        opinion_personas.append({"persona_id": pid, "profile": fallback_profile})

    if st_bar:
        st_bar.progress(
            0.15,
            text=f"뉴스 탐색 중... (선택 {len(selected_personas)}명 + 네트워크 {len(opinion_personas) - len(selected_personas)}명)",
        )
    print(
        f"[DEBUG] step1 inputs: opinion_personas={len(opinion_personas)}, "
        f"data_pool={len(data_pool)}"
    )

    # =========================================================================
    # Step 1: 뉴스 탐색 (페르소나당 1회 호출로 배치 결정)
    # =========================================================================
    if st_bar:
        st_bar.progress(0.30, text="뉴스 탐색 중... (배치)")
    exploration = run_news_exploration(
        opinion_personas,
        data_pool,
        max_workers=llm_max_workers,
    )

    # =========================================================================
    # Step 2: 의견 생성 (탐색에서 선택된 뉴스만 대상)
    # =========================================================================
    if st_bar:
        st_bar.progress(0.45, text="의견 생성 중...")
    opinions = generate_opinions(
        opinion_personas,
        exploration,
        data_pool,
        max_workers=llm_max_workers,
    )
    # =========================================================================

    filename = f"{timestamp}_{target_vcf if target_vcf else 'results'}_results_C.json"
    target_dir  = str(BASE_DIR / "data" / "output" / str(target_vcf) / llm.config.model_name / f"history_{history}" / f"{data}_{total_personas}" / "case_C")
    os.makedirs(target_dir, exist_ok=True)

    response_output_filename = os.path.join(target_dir, filename)
    output_filename = os.path.join(target_dir, f"{timestamp}_{target_vcf}_C.json")
    opinions_output_filename = os.path.join(target_dir, f"{timestamp}_{target_vcf}_opinions.json")
    

    with open(opinions_output_filename, "w", encoding="utf-8") as f:
        json.dump(opinions, f, ensure_ascii=False, indent=2)

    _get_twhin()

    if st_bar:
        st_bar.progress(0.50, text=f"{total_personas}명의 페르소나 응답을 생성합니다.")

    def run_pipeline(p):
        p_id = str(p["persona_id"])

        # 4. 피드 (이웃 content 수집 → bio 유사도 정렬)
        net_contents = collect_network_contents(p_id, proxy_data, opinions)
        feed = generate_feed(p_id, proxy_data, net_contents, top_k=feed_top_n)
        top_contents = [item["content"] for item in feed]
        # print("top contents", top_contents)

        # 5. 중요도 (피드 상위 N개 content 대상)
        importance_results = evaluate_context_importance(
            persona=p,
            selected_contents=top_contents,
            val_mode=val_mode,
        )

        # print("중요도 점수: ", importance_results)

        sorted_items = sorted(
            importance_results.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )
        top_thinkings = [
            {
                "thinking": item[1].get('thinking', ''),
                "score": item[1].get('score', 0)  # score 값이 없을 경우를 대비해 기본값 0 지정
            }
            for item in sorted_items[:K_VALUE]
        ]
        # top_thinkings = [item[1].get("thinking", "") for item in sorted_items[:K_VALUE]]

        top_ctx = {"event_keyword": {"keyword": event, "context": top_thinkings}}

        # 6. ask_persona
        res, _ = ask_persona_C(
            llm, p, query, options, event, top_ctx,
            prompt_templates, {"query_keyword": event}, K_VALUE, val_mode=val_mode, history=history
        )
        return p_id, res, importance_results

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_pipeline, p) for p in selected_personas]
        for i, f in enumerate(as_completed(futures)):
            p_id, res, imp_res = f.result()
            results[p_id] = {query: res}
            importance_history[p_id] = imp_res
            if st_bar:
                progress = 0.50 + (0.50 * (i + 1) / total_personas)
                st_bar.progress(
                    min(progress, 1.0),
                    text=f"시뮬레이션 진행 중: {i + 1}/{total_personas} 완료",
                )
                
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print("\n" + "="*40)
    print(f"총 소요 시간: {minutes}분 {seconds:.2f}초 ({elapsed_time:.2f}초)")
    print("="*40 + "\n")

    
    try:
        with open(response_output_filename, 'w', encoding='utf-8') as f:
            # results 딕셔너리를 통째로 저장
            json.dump(results, f, ensure_ascii=False, indent=4)
        print(f"\n[저장 완료] 페르소나별 응답 결과:\n{response_output_filename}")
    except Exception as e:
        print(f"파일 저장 중 에러 발생: {e}")

    # 중요도 저장
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(importance_history, f, ensure_ascii=False, indent=4)
    print(f"\n중요도 분석 결과 저장:\n{output_filename}")

    if st_bar:
        st_bar.progress(1.0, text="시뮬레이션 완료!")
    return results
