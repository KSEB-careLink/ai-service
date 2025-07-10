import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reminder(prompt: str, relation: str = "아버지"):
    system_prompt = (
        "너는 치매 초기 환자를 위한 회상 도우미야. "
        f"사용자는 환자의 {relation}이야. "
        f"생성되는 문장은 '{relation}, 오늘은~'처럼 자연스럽게 관계를 반영해서 시작해줘. "
        "말투는 따뜻하고 친근하게, 문장은 너무 길지 않게. "
        "말 끝은 '…'이나 쉼표 등을 이용해서 부드럽게 마무리되도록 만들어줘. "
        "예를 들어, '아버지, 오늘은 날씨가 참 좋았죠…'처럼 생성해줘. "
        "기억을 자극할 수 있는 소중한 순간이나 장소, 감정을 담아줘."
        "중간에 말을 좀 쉬었으면 좋겠어."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
