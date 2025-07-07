import os
import openai
from dotenv import load_dotenv

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reminder(prompt: str):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 치매 환자를 위한 회상 도우미야."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content