import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

<<<<<<< HEAD
def generate_reminder(prompt: str, relationship: str):
    # 관계 정보 포함
    system_content = (
        f"너는 경증 치매 환자의 기억을 상기시키고 정서적 안정을 돕는 회상 도우미야. "
        f"사용자가 가입할 때 제공한 정보 중, 환자와 보호자의 관계는 '{relationship}' 이야. "
        f"{relationship}의 말투와 어조를 반영해서 문장을 작성해야 해. "
        "너의 목표는 환자가 편안함을 느끼고, 과거의 긍정적인 경험이나 소중한 사람을 떠올리도록 돕는 것이야. "
        "문장은 짧고 이해하기 쉬워야 하며, 부드럽고 따뜻한 말투로 말해야 해. "
        "존댓말을 사용하고, 환자를 격려하거나 칭찬하며, 너무 많은 정보를 한 문장에 담지 말고 간단하게 나눠줘. "
        "말끝은 자연스럽게 이어지도록 하고, 환자가 기분 좋게 들을 수 있도록 배려해줘. "
        "마치 가족이 말하듯 편안한 느낌으로 작성해줘."
=======
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
>>>>>>> origin/main
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
<<<<<<< HEAD
            {"role": "system", "content": system_content},
=======
            {"role": "system", "content": system_prompt},
>>>>>>> origin/main
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
