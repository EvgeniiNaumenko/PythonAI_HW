import os
import platform
import subprocess
import random
import pyttsx3
import speech_recognition as sr
from datetime import datetime
import requests
import json
from dotenv import load_dotenv

from gtts import gTTS # озвучка
import tempfile

load_dotenv()

# ========== КЛЮЧ ДО GROQ ==========
GROQ_API_KEY = os.environ.get("GROQ_API_KEY") or ""
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set")

# ========== Глобальні налаштування ==========
MODEL = "llama-3.3-70b-versatile"  # Актуальний
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

VOICE_GENDER = "female"  # "female" або "male"
VOICE_VOLUME = 1.0  # 0.0 - 1.0
VOICE_RATE = 150  # Темп (звук/хв)

# ========== Команди-активатори ==========
JOKES = [
    "Йде студент по коридору, бачить — сесія. І ховається.",
    "Програміст — це машина для перетворення кави в код.",
    "Песиміст бачить темряву в тунелі, оптиміст — світло, інженер — потяг, машиніст — двох дурнів на рейках."
]

TIPS = [
    "Не забувай робити паузи під час роботи за комп’ютером.",
    "Пий більше води, твій мозок буде вдячний.",
    "Ніколи не бійся пробувати щось нове — так з’являється досвід."
]

# Лог файлу для збереження розмов
LOG_FILE = "jarvis_log.json"

COMMANDS = {
    "яка година": lambda: f"Зараз {datetime.now().strftime('%H:%M')}",
    "яка дата": lambda: f"Сьогодні {datetime.now().strftime('%d.%m.%Y')}",
    "анекдот": lambda: random.choice(JOKES),
    "порада дня": lambda: random.choice(TIPS),
}


# ========= Логування ==========
def log_conversation(user_text, assistant_text):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user": user_text,
        "assistant": assistant_text
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def show_history(n=3):
    # Зчитуємо останні n записів і повертаємо їх текстом
    if not os.path.exists(LOG_FILE):
        return "Історія розмов поки що порожня."

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    last_entries = lines[-n:]
    history_text = ""
    for line in last_entries:
        try:
            entry = json.loads(line)
            ts = entry.get("timestamp", "")
            user = entry.get("user", "")
            assistant = entry.get("assistant", "")
            history_text += f"\n[{ts}]\nТи: {user}\nДжарвіс: {assistant}\n"
        except:
            continue

    return history_text if history_text else "Історія розмов поки що порожня."


# ========== Озвучка ==========
def init_tts_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')

    print("\n🎙️ Доступні голоси:")
    for i, voice in enumerate(voices):
        print(f"{i}: {voice.name} - {voice.id}")

    # Встановлюємо голос за заданим VOICE_GENDER, якщо не знайшли, лишаємо дефолтний
    selected_voice_id = None
    for voice in voices:
        # Часткова перевірка по gender (можна зробити краще)
        name_lower = voice.name.lower()
        if VOICE_GENDER == "female" and "female" in name_lower:
            selected_voice_id = voice.id
            break
        elif VOICE_GENDER == "male" and "male" in name_lower:
            selected_voice_id = voice.id
            break

    if selected_voice_id:
        engine.setProperty('voice', selected_voice_id)

    engine.setProperty('volume', VOICE_VOLUME)
    engine.setProperty('rate', VOICE_RATE)
    return engine


def speak_ua(text):
    engine = init_tts_engine()
    engine.say(text)
    engine.runAndWait()

# def play_audio(path):
#     system = platform.system()
#     try:
#         if system == "Darwin":
#             subprocess.run(["afplay", path])
#         elif system == "Windows":
#             os.startfile(path)
#         else:
#             subprocess.run(["mpg123", path])
#     except Exception as e:
#         print(f"Error playing audio: {e}")
#
# def speak_ua(text):
#     if not text:
#         return
#     try:
#         tts = gTTS(text=text, lang="uk")
#         with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
#             path = fp.name
#         tts.save(path)
#         play_audio(path)
#         # close player
#         # kill process
#         os.remove(path)
#     except Exception as e:
#         print("Не вдалось озвучити!", e)


# Функція оновлення гучності в рамках запущеного движка (якщо потрібно)
def change_volume(new_volume):
    global VOICE_VOLUME
    VOICE_VOLUME = min(max(new_volume, 0.0), 1.0)
    print(f"🔊 Гучність встановлено на {VOICE_VOLUME}")
    # Можна додати логіку оновлення движка, якщо він в глобальній змінній


# ========== Вибір голосу ==========
def choose_voice_settings():
    global VOICE_GENDER, VOICE_VOLUME, VOICE_RATE

    print("\n🎙️ Обери голос:")
    print("1. 👩 Жіночий")
    print("2. 👨 Чоловічий")
    choice = input("Вибір (1/2): ").strip()
    VOICE_GENDER = "female" if choice == "1" else "male"

    vol = input("🔊 Введи гучність (0.0 до 1.0, за замовчуванням 1.0): ").strip()
    try:
        VOICE_VOLUME = min(max(float(vol), 0.0), 1.0)
    except:
        VOICE_VOLUME = 1.0

    rate = input("🚀 Введи швидкість мови (типово 150): ").strip()
    try:
        VOICE_RATE = int(rate)
    except:
        VOICE_RATE = 150


# ========== Розпізнавання голосу ==========
def choose_microphone():
    print("\n🎤 Доступні мікрофони:")
    for idx, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"{idx}: {name}")
    try:
        index = int(input("Введи номер мікрофона: "))
        return index
    except:
        print("❌ Невірний ввід. Використовується мікрофон за замовчуванням.")
        return None


def listen_ukrainian(device_index=None, timeout=5, phrase_time_limit=20):
    r = sr.Recognizer()
    with sr.Microphone(device_index=device_index) as source:
        print("🎧 Говори щось українською...")
        r.adjust_for_ambient_noise(source, duration=0.8)
        try:
            audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            text = r.recognize_google(audio, language="uk-UA")
            print("🗣️ Ти сказав(ла):", text)
            return text
        except sr.WaitTimeoutError:
            print("⌛ Тайм-аут: нічого не почуто.")
        except Exception as err:
            print("❌ Помилка розпізнавання:", err)
    return None


# ========== Запит до GROQ ==========
def ask_groq(prompt, history=None):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {"role": "system", "content": "Ти — розумний, доброзичливий український асистент. Відповідай чітко й просто."}
    ]

    if history:
        messages += history[-10:]  # останні 10 повідомлень

    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }

    try:
        res = requests.post(GROQ_URL, headers=headers, json=payload)
        if res.status_code != 200:
            print(f"❌ Помилка API: {res.status_code} {res.text}")
            return "Вибач, сталася помилка при зверненні до ШІ."
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception as err:
        print("❌ Запит не вдалось виконати:", err)
        return "На жаль, не вдалося отримати відповідь від ШІ."


# ========== Головна функція ==========
def main():
    print("👋 Привіт! Я Джарвіс — твій голосовий помічник.")
    choose_voice_settings()
    mic_index = choose_microphone()
    conversation_history = []

    try:
        while True:
            query = listen_ukrainian(device_index=mic_index)
            if not query:
                continue

            query_l = query.lower().strip()

            # Вихід
            if query_l in ["вийти", "дякую", "завершити", "стоп"]:
                print("👋 Бувай!")
                speak_ua("Бувай!")
                break

            # Команда показати історію
            if query_l == "покажи історію":
                history_text = show_history(n=3)
                print(history_text)
                speak_ua(history_text)
                continue

            # Команда зміни гучності (наприклад: "гучність 0.7")
            if query_l.startswith("гучність"):
                parts = query_l.split()
                if len(parts) == 2:
                    try:
                        vol = float(parts[1])
                        change_volume(vol)
                        speak_ua(f"Гучність встановлено на {vol}")
                    except ValueError:
                        speak_ua("Невірне значення гучності.")
                else:
                    speak_ua("Скажи 'гучність' та число від 0 до 1.")
                continue

            # Інші команди
            if query_l in COMMANDS:
                answer = COMMANDS[query_l]()
                print("🧭 Команда:", query_l)
                print("📤 Відповідь:", answer)
                speak_ua(answer)
                # Логування
                log_conversation(query, answer)
                continue

            # Запит до ШІ
            print("🤖 Думаю...")
            answer = ask_groq(query, history=conversation_history)
            print("📤 Відповідь:", answer)

            conversation_history.append({"role": "user", "content": query})
            conversation_history.append({"role": "assistant", "content": answer})
            conversation_history = conversation_history[-20:]

            speak_ua(answer)
            # Логування
            log_conversation(query, answer)

    except KeyboardInterrupt:
        print("\n🛑 Вихід через Ctrl+C. До зустрічі!")
        speak_ua("До зустрічі!")



# ========== Точка входу ==========
if __name__ == "__main__":
    main()
