import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

# 파일 경로 및 환경 변수 설정
BASE_DIR = Path(__file__).resolve().parents[2] 
env_path = BASE_DIR / ".env"
load_dotenv(env_path)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def search_events(keyword, num, start_date, end_date, master_results, seen_links):
    """
    특정 키워드에 대해 수집을 수행하고 master_results에 통합함
    """
    per_page = 20 
    current_page = 1
    collected_for_this_keyword = 0
    
    try:
        start_obj = datetime.strptime(start_date, "%Y-%m-%d")
        end_obj = datetime.strptime(end_date, "%Y-%m-%d")
        tbs_string = f"cdr:1,cd_min:{start_obj.strftime('%m/%d/%Y')},cd_max:{end_obj.strftime('%m/%d/%Y')}"
    except ValueError:
        raise ValueError("날짜 형식 오류")

    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }

    print(f"\n🚀 키워드 시작: [{keyword}]")

    while collected_for_this_keyword < num:
        print(f"  - {current_page}페이지 수집 중... (현재 키워드 누적: {collected_for_this_keyword}개)")
        
        payload = json.dumps({
            "q": keyword,
            "num": per_page,
            "page": current_page,
            "gl": "us",
            "hl": "en",
            "tbs": tbs_string
        })

        try:
            response = requests.post("https://google.serper.dev/search", headers=headers, data=payload)
            if response.status_code != 200:
                print(f"  ❌ 에러 발생: {response.status_code}")
                break

            data = response.json()
            organic = data.get("organic", [])

            if not organic:
                print("  ⚠️ 더 이상의 검색 결과가 없습니다.")
                break

            for item in organic:
                title = item.get("title", "No Title")
                link = item.get("link", "")
                date = item.get("date", "No Date")
                snippet = item.get("snippet", "")

                # 💡 중복 체크 (Link 기준)
                if link in seen_links:
                    continue
                
                # 중복이 아니면 추가
                seen_links.add(link)
                
                # 제목 중복 방지 (딕셔너리 키용)
                dict_key = title
                if dict_key in master_results:
                    dict_key = f"{title}_{len(master_results)}"
                
                master_results[dict_key] = {
                    "link": link,
                    "description": snippet,
                    "date": date,
                    "source_keyword": keyword # 어떤 키워드로 수집되었는지 기록 (선택 사항)
                }

                collected_for_this_keyword += 1
                
                # 키워드별 목표치 달성 시 종료
                if collected_for_this_keyword >= num:
                    break
            
            # 다음 페이지로
            current_page += 1
            if current_page > 100: # 구글 제한
                break
                
        except Exception as e:
            print(f"  ❌ 호출 중 오류: {e}")
            break

    return master_results, seen_links


if __name__ == "__main__":     
    if not SERPER_API_KEY:
        raise ValueError(".env 파일에 SERPER_API_KEY가 없습니다.")
    
    # 💡 수집할 키워드 리스트
    keywords = [
        "Biden-Harris border crisis costs American workers jobs",
        "Biden immigration disaster hurts American workers",
        "Biden open borders lower wages American workers",
    ]
    
    total_results_per_keyword = 70
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # 전체 결과를 담을 그릇
    master_dict = {}
    seen_links = set() # URL 중복 체크용 셋
    
    # 키워드 루프 실행
    for kw in keywords:
        master_dict, seen_links = search_events(
            keyword=kw, 
            num=total_results_per_keyword, 
            start_date=start_date,
            end_date=end_date,
            master_results=master_dict,
            seen_links=seen_links
        )
    
    # 결과 저장
    output_dir = str(BASE_DIR / "data" / "external")
    os.makedirs(output_dir, exist_ok=True)
    filename = "data_VCF9223_0427_2.json"
    save_path = os.path.join(output_dir, filename)

    print(f"\n✅ 전체 수집 종료. 중복 제거 후 최종 뉴스 수: {len(master_dict)}")

    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(master_dict, f, ensure_ascii=False, indent=4)
        print(f"성공적으로 저장되었습니다. 경로: {save_path}")
    except Exception as e:
        print(f"파일 저장 중 오류 발생: {e}")