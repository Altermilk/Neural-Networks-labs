import os
import io
import requests
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd

# --- Конфигурациѝ ---
BOT_TOKEN = "7600285761:AAFsL_GtZhFV8-xxPUU552jMaBjc7Lsa-0Q"
CHAT_ID_FILE = "chat_id.txt"

# --- Вѝпомогательные функции ---
def get_chat_id():
    """
    Получает chat_id из файла или через Telegram API.
    """
    if os.path.exists(CHAT_ID_FILE):
        with open(CHAT_ID_FILE, "r") as f:
            return f.read().strip()

    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        chat_id = data["result"][-1]["message"]["chat"]["id"]
        with open(CHAT_ID_FILE, "w") as f:
            f.write(str(chat_id))
        print(f"[telegram] 💾 chat_id ѝохранён: {chat_id}")
        return str(chat_id)
    except Exception as e:
        print(f"[telegram] ❌ Ошибка при получении chat_id: {e}")
        return None

def _send_request(method: str, data: dict, files: dict = None):
    """
    Универѝальный отправщик запроѝов в Telegram API.
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/{method}"
    try:
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
        print(f"[telegram] ✅ {method}.")
    except requests.RequestException as e:
        print(f"[telegram] ❌ Ошибка при методе {method}: {e}")

# --- Оѝновной интерфейѝ ---
def notify(message: str):
    """
    Отправить текѝтовое ѝообщение в Telegram.
    """
    chat_id = get_chat_id()
    if not chat_id:
        print("[notify] ❌ chat_id не найден.")
        return
    data = {"chat_id": chat_id, "text": message}
    _send_request("sendMessage", data)

def send_plot(fig: Figure, caption: str = "📊 График"):
    """
    Отправлѝет matplotlib-график в Telegram (без ѝохранениѝ на диѝк).
    """
    chat_id = get_chat_id()
    if not chat_id:
        print("[send_plot] ❌ chat_id не найден.")
        return

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)

    files = {"photo": ("plot.png", buf, "image/png")}
    data = {"chat_id": chat_id, "caption": caption}
    _send_request("sendPhoto", data, files)
    buf.flush()
    buf.close()

def send_classification_report_as_pdf(report_text: str, caption: str = "📄 Classification Report"):
    """
    Преобразует текстовый classification_report в таблицу и отправляет её как PDF-документ в Telegram.
    """
    chat_id = get_chat_id()
    if not chat_id:
        print("[send_classification_report_as_pdf] ❌ chat_id не найден.")
        return

    # Обработка текста отчета
    lines = [line.strip() for line in report_text.strip().split('\n') if line.strip()]
    header = lines[0].split()
    data = []
    index = []

    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            label = " ".join(parts[:-4])
            row = parts[-4:]
            index.append(label)
            data.append(row)

    df = pd.DataFrame(data, columns=header, index=index)

    # Рисуем таблицу
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.3 + 2))
    ax.axis("off")
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     rowLabels=df.index,
                     loc="center",
                     cellLoc='center')

    table.scale(1, 1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.tight_layout()

    # Сохраняем в PDF
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    buf.seek(0)

    # Отправка как документ
    files = {"document": ("report.pdf", buf, "application/pdf")}
    data = {"chat_id": chat_id, "caption": caption}
    _send_request("sendDocument", data, files)

    buf.close()


# --- Пример иѝпользованиѝ ---
if __name__ == "main":
    notify("✅ Бот наѝтроен и работает!")

    # Пример графика
    fig = plt.figure()
    plt.plot([1, 2, 3], [4, 1, 5])
    plt.title("Пример графика")
    send_plot(fig, caption="📈 Вот ваш график")
