import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Імпортуємо функцію аналізу з нашого першого файлу
from scoring_engine import calculate_ats_metrics

# Завантажуємо змінні оточення (токен від Google ховаємо в .env)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("❌ КРИТИЧНА ПОМИЛКА: Ключ GOOGLE_API_KEY не знайдено у файлі .env або змінних оточення!")
client = genai.Client(api_key=api_key)


def get_ai_coaching_report(scoring_payload: dict) -> dict:
    """
    Приймає готові метрики, надсилає в Gemini
    і повертає структуровані поради як Python-словник.
    """
    input_data_str = json.dumps(scoring_payload, indent=2, ensure_ascii=False)

    prompt = f"""
    Ти — професійний IT-рекрутер та кар'єрний коуч (AI Resume Coach). 
    Твоє завдання — проаналізувати математичні метрики відповідності резюме до вакансії та дати кандидату конструктивні поради.

    Ось аналітичні дані від нашої системи (JSON):
    {input_data_str}

    Контекст для твоєї оцінки:
    - Бал (ats_score) розрахований за допомогою TF-IDF та Cosine Similarity. 
    - Зона 55%–75% є ОПТИМАЛЬНОЮ (це сильний кандидат, не вважайте це низьким балом!).
    - Якщо бал менший за 55%, кандидату дійсно бракує важливих слів чи досвіду.

    Твоє завдання — згенерувати відповідь СУВОРO у форматі JSON за такою схемою:
    {{
      "verdict": "Короткий загальний висновок про сумісність (1 речення)",
      "strengths": ["Дві сильні сторони резюме щодо цієї вакансії"],
      "improvements": [
        {{
          "section": "Назва розділу CV, який треба покращити (наприклад, Professional Experience чи Skills)",
          "original_text": "Конкретна слабка або занадто загальна фраза з CV кандидата",
          "suggested_text": "Твій покращений варіант цієї фрази (додай активні дієслова, натякни на метрики)",
          "reason": "Чому твій варіант краще зачепить рекрутера або алгоритм"
        }}
      ]
    }}

    Правила: 
    1. Відповідь має бути виключно у форматі JSON, без жодних вступних фраз, привітань чи markdown-обгорток (```json).
    2. Пиши мовою резюме кандидата, але поля verdict та reason мають бути виключно українською!
    """

    for attempt in range(3):  # Робимо максимум 3 спроби
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3 #низька температура для професійнішого тону без художніх прикрас і вигадок
                )
            )
            return json.loads(response.text)
            
        except Exception as e:
            error_msg = str(e)
            # Якщо помилка 503 і це ще не остання спроба - чекаємо і повторюємо
            if "503" in error_msg and attempt < 2:
                print(f"Gemini API перевантажено (503). Спроба {attempt + 1} з 3. Пауза 2 сек...")
                time.sleep(2)
                continue  # Йдемо на наступне коло циклу
            
            # Якщо це інша помилка (наприклад, 400) або спроби закінчилися:
            print(f"Помилка Gemini API: {error_msg}")
            # Повертаємо безпечну заглушку, щоб фронтенд не впав і показав хоч щось
            return {
                "verdict": "ШІ-сервери Google зараз недоступні. Ми проаналізували ваше резюме математично (див. ATS Score), але для текстових порад спробуйте ще раз за хвилину.",
                "strengths": [],
                "improvements": []
            }


def run_full_pipeline(job_text: str, cv_pdf_bytes: bytes) -> dict:
    """
    УНІВЕРСАЛЬНИЙ ЛАНЦЮЖОК:
    Поєднує математику та ШІ. Саме її виконавець викликатиме у FastAPI.
    """
    # Етап 1: Рахуємо метрики
    scoring_payload = calculate_ats_metrics(job_text, cv_pdf_bytes)

    # Етап 2: Запитуємо ШІ
    ai_report = get_ai_coaching_report(scoring_payload)

    # Етап 3: Формуємо фінальний "JSON" для фронтенду
    final_response = {
        "ats_score": min(scoring_payload["ats_score"] + 20.0, 100.0), # +20 для "гарнішого" балу на фронтенді, вище 100 не буде z
        "missing_skills": scoring_payload["missing_skills"],
        "warning": scoring_payload["warning"],
        "ai_coaching": ai_report
    }
    return final_response
