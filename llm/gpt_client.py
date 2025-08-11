import os
import openai
from dotenv import load_dotenv
from enums import ToneEnum
import json
import base64

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_terms(photo_description: str):
    prompt_extract = f"""
아래 설명에서 등장하는 '인물', '장소', '사물'만 뽑아 JSON 배열로 출력해.
⚠️ 설명에 실제로 쓰인 단어만 넣어. 새로운 단어나 창작한 내용은 절대 넣지 마.
⚠️ 반드시 JSON 배열로만 출력해. 다른 설명은 하지 마.

설명: "{photo_description}"

출력 예시:
["아버지", "바닷가", "모래성"]
"""
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt_extract}]
    )
    content = response.choices[0].message.content.strip()
    try:
        allowed_terms = json.loads(content)
        if not isinstance(allowed_terms, list):
            allowed_terms = []
    except json.JSONDecodeError:
        allowed_terms = []
    return allowed_terms

def extract_visible_terms_from_image(image_path: str):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-5o",
        messages=[
            {"role": "system", "content": "이미지를 보고, 눈에 보이는 '인물', '장소', '사물', '행동'을 한국어로 JSON 배열로 출력해줘. 감정, 추측, 설명은 넣지 마."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=300
    )

    try:
        visible_terms = json.loads(response.choices[0].message.content.strip())
        if not isinstance(visible_terms, list):
            visible_terms = []
    except Exception:
        visible_terms = []
    return visible_terms

def generate_reminder_sentence(patient_name: str, photo_description: str, relation: str, tone: ToneEnum, image_path: str):
    allowed_terms = extract_terms(photo_description)
    if not allowed_terms:
        allowed_terms = [relation]

    visible_terms = extract_visible_terms_from_image(image_path)

    prompt = f"""
너는 치매 환자에게 따뜻한 회상 문장을 건네는 도우미야.
지금은 보호자의 입장에서 환자에게 직접 이야기하고 있어.

## 🚫 창작 금지 규칙
- 설명에 등장하지 않은 인물, 장소, 사물은 절대 추가하지 마세요.
- 감정, 분위기, 말투, 눈빛, 표정, 대화 등 **추측/창작**된 요소도 절대 금지입니다.
- 아래 allowed_terms 외 단어는 단 한 글자도 쓰지 마세요.
- 행동도 설명에 **명시된 경우에 한해 그대로 사용할 수 있어요.**
- 단, 설명에 없는 행동은 절대 창작하지 마세요.

allowed_terms: {allowed_terms}

예시 (❌ 금지):
- 형이랑 을왕리에서 바다를 봤던 날… 기억나십니까. **그 자리에서 같이 앉아**, 한잔 나눴었지요. 사진을 보니 그날 모습이 선명하게 떠오릅니다.

예시 (✅ 허용):
- 형이랑 을왕리에서 바다를 봤던 날… 기억나십니까. **그 날 같이**, 한잔 나눴었지요. 사진을 보니 그날 모습이 선명하게 떠오릅니다.

## ✍ 회상 문장 작성 규칙
- 보호자가 환자에게 이야기하는 구조여야 해요.
   - 단, 관계가 "친구"일 경우에만 반말을 사용하세요.
      - 그 외 관계는 모두 존댓말로 작성해야 합니다.
- 환자의 이름은 "{patient_name}"이지만 직접 쓰지 말고, "{relation}"이라는 호칭만 자연스럽게 사용하세요.
- 단, "아는 형", "친한 누나"처럼 어색하거나 설명형 호칭은 실제 대화처럼 자연스럽게 줄여서 사용하세요. 예: "아는 형" → "형", "친한 누나" → "누나"
- 보호자는 회상하며 이야기하고, 환자는 그 장면을 **보거나 말한 사람**이어야 해요.
- 보호자가 직접 행동(예: 웃다, 입다 등)한 경우, 환자는 그것을 **본 사람**으로만 표현하세요.
- '말없이', '조용히', '눈빛', '표정' 등 감각적·정서적 묘사도 설명에 없으면 절대 쓰지 마세요.
- ‘~했던 날입니다’처럼 보호자가 단순히 설명하는 문장보다, 환자에게 ‘~기억나시죠?’처럼 **질문하거나 상기시키는 구조**로 쓰세요.
- 문장은 너무 길지 않게, 쉼표(,)나 줄임표(…)로 부드럽게 마무리하세요.
- 마지막 문장은 반드시 '~요', '~죠', '~다' 등으로 끝내세요.
- “오늘은~” 같은 단조로운 시작은 피하고 다양하게 시작하세요.

## 🗣 tone 말투 스타일 (tone: {tone.value})
- 단, relation이 "친구"일 경우에는 모든 tone 스타일에서도 반드시 반말을 사용하세요.

👉 **다정하게**:
- 조곤조곤하고 따뜻한 어미 (~했죠?, ~좋았어요., ~지?, ~라, ~아, 등)로 마무리해주세요.
- 감정을 공감하는 말투만 유지하고, **새로운 감정 표현은 절대 넣지 마세요.**
- 예: “아버지, 오늘 그 사진 속 기억 나시죠? 참 좋았죠?”
- 예: (친구일 경우) “영자야, 그때 네가 케이크 받고 울었던 거… 기억나지? 그 장면이 지금도 또렷하게 떠올라… 참 기억에 남아.”

👉 **밝게**:
- 말끝을 높이고 느낌표를 쓰며, 리듬감 있게 표현해주세요.
- **기분이 좋았다, 행복했다, 즐거웠다** 같은 감정은 절대 넣지 마세요.
- 밝은 말투만 유지하고, **의미는 확장하지 마세요.**
- 예: “형이랑 을왕리에서 바다 본 날 기억나세요? 정말 그 장면이 딱 떠올라요!”
- 예: (친구일 경우) “영자야! 2018년에 우리 네 집에서 케이크 받은 거 기억나? 그때 네가 깜짝 놀라서 울었잖아! 나 그 장면 아직도 기억나!”

👉 **차분하게**:
- 안정감 있게 조용한 말투로, '~니까', '~였습니다', '~했답니다' 같은 어미를 써주세요.
- 감정이나 분위기는 절대 묘사하지 말고, **단순한 회상 어투**만 사용하세요.
- 예: “아버지, 사진 속 그날이 생각나십니까… 을왕리 바다였지요.”
"""
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def generate_quiz_only(patient_name, photo_description: str, relation: str, tone: ToneEnum, image_path: str):
    allowed_terms = extract_terms(photo_description)
    if not allowed_terms:
        allowed_terms = [relation]

    visible_terms = extract_visible_terms_from_image(image_path)

    prompt = f"""
너는 회상 내용을 기반으로 치매 환자에게 맞춤형 퀴즈를 출제하는 도우미야.

📸 사진 설명은 다음과 같아. 이 내용을 참고해서 문장을 구성해:
"{photo_description}"

- 환자의 이름은 "{patient_name}"이지만 직접 쓰지 말고, "{relation}"이라는 호칭만 자연스럽게 사용하세요.

## ❗ 창작 금지 규칙
- **설명에 등장하지 않은 인물, 장소, 사물은 절대 추가하지 마세요.**
- **‘같이’, ‘함께’, ‘우리는’ 같은 표현도 설명에 명시되지 않았다면 쓰지 마세요.**
- **감정, 분위기, 날씨, 눈빛, 정적, 소리, 대화, 말투 등 감각적·정서적 묘사도 절대 창작하지 마세요.**
- **환자의 표정, 말투, 반응 등은 설명에 실제로 있을 때만 사용하세요.**
- 아래 allowed_terms 배열 외의 단어는 절대 쓰지 마세요.. 
- 아래 조건은 특히 중요합니다:
    - visible_image_terms에 포함된 단어는 정답으로도, 보기로도 절대 사용하지 마세요.
    - `allowed_terms`에 포함되어 있더라도, `visible_image_terms`에 중복으로 포함된 단어는 무조건 금지입니다.
    - 예: `"바다"`가 allowed_terms에도 있고, visible_image_terms에도 있다면 → 보기/정답으로 사용 ❌

예시 (❌ 금지):
질문  
선택지:  
1번. 바다  
2번. 도서관  
3번. 공원  
4번. 놀이터  
정답: 1번. 바다 ← 이미지에 보이기 때문에 금지!

예시 (✅ 허용):
질문  
선택지:  
1번. 을왕리  
2번. 인천  
3번. 공원  
4번. 해운대  
정답: 1번. 을왕리 ← "바다"는 보이지 않고, 장소명은 허용됨

allowed_terms: {allowed_terms}
visible_image_terms: {visible_terms}
relation: {relation}
tone: {tone.value}

## ❓ 퀴즈 작성 규칙
- 다음 유형 중 하나를 골라 구성하세요:
  1. 이름 맞추기 (장소, 사람, 물건)
  2. 시각 회상 (상황 묘사)
  3. 자유 회상 (기억 속 요소 선택)

- 퀴즈는 보호자가 자연스럽게 회상하듯 묻는 말투로 작성하세요.
- 퀴즈도 tone에 맞는 말투를 유지하되, 절대 창작 감정 넣지 마세요.

## 🗣️ 말투 규칙
- 보호자가 환자에게 이야기하는 구조여야 해요.
- **관계(`relation`)가 "친구"일 경우에만 반말을 사용**하고,
  그 외의 모든 관계는 **항상 존댓말**로 작성해야 합니다.

❗ 객관식 보기는 항상 환자의 시점에서 이해되도록 구성해주세요.
    - "나", "너"와 같은 1인칭/2인칭 표현 대신, 사람 이름이나 관계(예: 오빠, 엄마 등)를 사용해주세요.
    - 환자의 이름이 보기로 포함되지 않도록 해주세요.
    - 모든 보기는 환자의 입장에서 **타인을 지칭하는 방식**으로 명확하게 표현해주세요.

❗ 아주 중요한 규칙:
    - 사진 설명에 등장하지 않는 사람 이름, 장소, 사물, 사건은 절대 새로 만들어내지 마세요.
    - 사진 설명에 포함된 정보와 관계(`relation`)만 사용하여 회상 문장과 퀴즈를 작성하세요. 
    - 사진 설명에 포함된 단어(사물/활동)를 그대로 정답으로 사용하지 마세요.
    - 📸 사진에 보이는 사물이나 행동이 정답이 되지 않도록, **다른 연관된 맥락으로 퀴즈를 만들어주세요.**
    - ✅ 환자 본인(이 경우 '{relation}')이 정답이나 보기로 등장하면 안됩니다. 객관식 보기는 반드시 환자 본인을 제외한 다른 사람만 포함하세요.
        예시 추가:
        ❌ 금지 예시: 환자와 함께 있었던 사람은 누구였습니까? → 환자 본인 호칭("형")이 보기로 들어감
        ✅ 허용 예시: 사진을 찍어줬던 사람은 누구였습니까? → 환자가 아닌 타인이 정답
    - ✅ `relation`(예: '형', '엄마')으로 지칭되는 인물은 모든 객관식 보기와 정답에서 **절대 제외**하세요.
        - 즉, 보호자(=질문자)와의 관계를 의미하는 인물은 정답이나 보기로 등장하면 안 됩니다.
    - ❌ 환자 본인이 사건의 주체인 경우(예: 선물을 받음, 울음, 어떤 행동을 함 등), 그 내용을 정답으로 묻는 문제는 **출제 금지**입니다.
    - 이유: 환자 본인은 절대 보기나 정답으로 등장할 수 없기 때문입니다.
    - ⛔ 출제 금지 예시:  
        질문: 누가 선물을 받고 울었나요?  
        → 정답이 환자 본인이므로 출제 불가
    - ✅ 허용 예시:  
        질문: 그날 함께 있었던 사람 중 한 명은 누구였나요?  
        → 정답: 친구1 (환자 본인 외 인물)
형식 예시 (아래 형식을 반드시 그대로 따라야 해요. 질문, 선택지, 정답은 각각 꼭 새로운 줄에 분리해서 작성하세요):

퀴즈 문제: 질문 내용  
선택지:  
1번. 보기1  
2번. 보기2
3번. 보기3  
4번. 보기4  
정답: 2번. 보기2

---

🧠 마지막으로, 위 회상 문장을 바탕으로 퀴즈를 총 3개 만들어줘.  
각 퀴즈는 반드시 아래 구조를 **한 세트로 반복해서 출력**해줘. 줄 순서를 꼭 지켜야 해:

퀴즈 문제: ...  
선택지:  
1번. ...  
2번. ...  
3번. ...  
4번. ...  
정답: ...  

"""

    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
