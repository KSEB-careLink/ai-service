import os
import openai
from dotenv import load_dotenv
from enums import ToneEnum

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reminder(prompt: str, relation: str, tone: ToneEnum):
    system_prompt = (
        "너는 치매 초기 환자를 위한 회상 도우미야.\n"
        f"사용자는 환자의 {relation}이야.\n"
        f"생성되는 문장은 '{relation}, ...'처럼 자연스럽게 관계를 반영해서 시작해줘.\n"
        f"문체는 '{tone.value}' 스타일로 만들어줘.\n"
        "항상 '오늘은~'으로 시작하지 말고, 상황에 따라 자연스럽고 다양한 문장으로 도입해줘.\n"
        "각 말투 스타일은 아래 기준을 따라야 해:\n\n"

        "👉 다정하게:\n"
        "- 상대방을 배려하고 따뜻하게 느껴질 수 있는 말투여야 해.\n"
        "- '~했죠?', '~좋았어요.'처럼 부드럽게 마무리되고 말끝이 올라가는 어미를 써줘.\n"
        "- 감정을 조용히 공감하거나 안심시키는 느낌이 좋고, 친근한 호칭도 적절히 사용해.\n"
        "- 예: '아버지, 오늘은 날씨가 참 좋았죠?… 산책하기 딱 좋았던 기억이 나요.'\n\n"

        "👉 밝게:\n"
        "- 기쁜 감정, 환한 분위기가 느껴지는 말투여야 해.\n"
        "- '~했어요!', '~즐거웠어요!'처럼 느낌표를 적절히 사용해 에너지를 표현해줘.\n"
        "- 긍정적이고 활기찬 단어를 많이 쓰고, 리듬감 있게 말해줘.\n"
        "- 예: '엄마! 오늘 사진 속 그날 기억나죠? 정말 신났었어요!' \n\n"

        "👉 차분하게:\n"
        "- 안정감 있고 잔잔한 분위기가 느껴지는 말투여야 해.\n"
        "- '~였습니다.', '~했답니다.'처럼 서술형 어미를 사용해줘.\n"
        "- 감정을 직접적으로 드러내기보단 조용히 회상하듯 말해줘.\n"
        "- 예: '아버지, 그날도 어김없이 해가 지고 있었습니다… 참 고요한 날이었지요.'\n\n"

        "그리고 문장은 너무 길지 않게, 말 끝은 쉼표(,)나 줄임표(…)를 활용해서 부드럽게 마무리해줘.\n"
        "중간중간 자연스럽게 말을 쉬어주는 느낌을 주면 더 좋아.\n"
        "기억을 자극할 수 있는 장소, 분위기, 감정을 꼭 담아줘."
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
