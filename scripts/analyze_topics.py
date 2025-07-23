import json
import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.normpath(os.path.join(base_dir, '..', 'data', 'dataset.json'))
output_path = os.path.normpath(os.path.join(base_dir, '..', 'models', 'question_topics.json'))

with open(dataset_path, 'r', encoding='utf-8') as f:
    dataset = json.load(f)

id_to_topic = {}

def classify_topic(question_text: str) -> str:
    prompt = f"""
당신은 문장의 주제를 아래 5개의 카테고리 중 하나로 분류하는 전문가입니다.
문장에 등장하는 단어가 혼합되어 있어도, 반드시 가장 적합한 카테고리를 선택해야 합니다.

카테고리 정의:
1. 가족: 엄마, 아빠, 할머니, 할아버지, 이모, 삼촌 등 가족과 함께한 일반적인 기억
2. 여행: 여행지(해변, 산, 명소, 관광지 등)에서 있었던 기억
3. 학창시절: 학교, 교실, 체육대회, 방과 후 활동 등 학창시절에 관한 기억
4. 동네: 집 근처, 마당, 골목길, 동네 약수터 등 지역적인 생활 공간에 대한 기억
5. 환자가 제일 행복했던 기억: 위의 네 가지에 명확히 포함되지 않고, 환자가 개인적으로 가장 행복했다고 느낀 순간이나 상징적인 장면.  
   👉 **중요:** 문장 안에 가족 호칭이 들어있더라도, 그 장면이 “특별히 행복했던 기억”으로 강조된다면 이 카테고리를 선택해야 합니다.

⚠️ 반드시 위 5개 중 하나만 출력하세요. 다른 단어나 설명은 절대 쓰지 마세요.  
⚠️ 단순히 엄마, 아빠가 나온다고 ‘가족’으로 하지 말고, 문장에서 “특별히 행복했던 순간”으로 강조된 경우는 “환자가 제일 행복했던 기억”으로 분류하세요.

이제 아래 문장을 분류하세요:
"{question_text}"
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        topic = response.choices[0].message.content.strip()
        return topic
    except Exception as e:
        print("❌ LLM 호출 오류:", e)
        return "기타"

for item in dataset:
    q_id = item["question_id"]
    q_text = item["question"]
    topic = classify_topic(q_text)
    print(f"ID {q_id} → {topic}")
    id_to_topic[q_id] = topic

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(id_to_topic, f, ensure_ascii=False, indent=2)

print(f"✅ 주제 매핑 저장 완료: {output_path}")
