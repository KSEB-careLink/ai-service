import os
import openai
from dotenv import load_dotenv
from enums import ToneEnum

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reminder(patient_name: str, photo_description: str, relation: str, tone: ToneEnum):
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
        "- 예: '엄마! 오늘 사진 속 그날 기억나죠? 정말 신났었어요!'\n\n"

        "👉 차분하게:\n"
        "- 안정감 있고 잔잔한 분위기가 느껴지는 말투여야 해.\n"
        "- '~였습니다.', '~했답니다.'처럼 서술형 어미를 사용해줘.\n"
        "- 감정을 직접적으로 드러내기보단 조용히 회상하듯 말해줘.\n"
        "- 예: '아버지, 그날도 어김없이 해가 지고 있었습니다… 참 고요한 날이었지요.'\n\n"

        "그리고 문장은 너무 길지 않게, 말 끝은 쉼표(,)나 줄임표(…)를 활용해서 부드럽게 마무리해줘.\n"
        "중간중간 자연스럽게 말을 쉬어주는 느낌을 주면 더 좋아.\n"
        "기억을 자극할 수 있는 장소, 분위기, 감정을 꼭 담아줘."
    )

    # ✅ prompt를 system_prompt 밖에서 작성
    prompt = f"""
    - 환자 이름: {patient_name}
    - 사진 설명: {photo_description}
    - 보호자와의 관계: {relation}
    - 보호자의 말투: {tone}

    퀴즈 유형은 다음 중 하나를 자동으로 선택해서 생성해주세요:
    1. 이름 맞추기 – 사진 속 사람, 장소, 물건의 이름을 맞추는 문제
    2. 시각 회상 – 사진 배경이나 상황을 설명하고 기억을 유도하는 문제
    3. 자유 회상 – 사진을 보고 떠오를 수 있는 기억을 기반으로 보기 4개를 주고, 가장 관련 있는 것을 고르는 **객관식 퀴즈**로 만들어주세요

    환자가 이해하기 쉽게 다정하고 천천히 말하는 어조로 구성해주세요.

    출력 형식:
    회상 문장: ...
    퀴즈 유형: ...
    퀴즈 문제: ...
    선택지: 1번, 2번, 3번, 4번
    정답: ...
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
