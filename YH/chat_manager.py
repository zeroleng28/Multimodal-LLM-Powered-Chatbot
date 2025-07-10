# Refactored chat_manager.py to Align with Updated Code
import json
import os
from datetime import datetime

CHAT_HISTORY_DIR = "chat_histories"

# Ensure that the directory for storing chat histories exists
def ensure_directory_exists():
    if not os.path.exists(CHAT_HISTORY_DIR):
        os.makedirs(CHAT_HISTORY_DIR)

# Generate a new filename for a chat history based on timestamp
def generate_chat_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp}.json"
    return os.path.join(CHAT_HISTORY_DIR, filename)

# Initialize a new conversation, optionally storing multiple chunk types
def init_new_conversation(pdf_chunks=None, image_chunks=None, audio_chunks=None):
    ensure_directory_exists()
    filename = generate_chat_filename()
    data = {
        "messages": [],
        "pdf_chunks": pdf_chunks if pdf_chunks else [],
        "image_chunks": image_chunks if image_chunks else [],
        "audio_chunks": audio_chunks if audio_chunks else []
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename

# Save chat history with support for multiple chunk types
def save_chat_history(chat_history, filename, pdf_chunks=None, image_chunks=None, audio_chunks=None):
    ensure_directory_exists()
    existing_data = {}
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing_data = {}

    existing_data["messages"] = chat_history

    if pdf_chunks is not None:
        existing_data["pdf_chunks"] = pdf_chunks
    if image_chunks is not None:
        existing_data["image_chunks"] = image_chunks
    if audio_chunks is not None:
        existing_data["audio_chunks"] = audio_chunks

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

# Load chat history with support for multiple chunk types
def load_chat_history(filename):
    if not os.path.exists(filename):
        return [], [], [], []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
            messages = data.get("messages", [])
            pdf_chunks = data.get("pdf_chunks", [])
            image_chunks = data.get("image_chunks", [])
            audio_chunks = data.get("audio_chunks", [])
            return messages, pdf_chunks, image_chunks, audio_chunks
    except (json.JSONDecodeError, IOError):
        return [], [], [], []

# List all available chat histories
def list_chat_histories():
    ensure_directory_exists()
    files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(CHAT_HISTORY_DIR, x)), reverse=True)
    history_list = []
    for f in files:
        try:
            timestamp_str = f.replace("chat_", "").replace(".json", "")
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            display_name = dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            display_name = f
        history_list.append((os.path.join(CHAT_HISTORY_DIR, f), display_name))
    return history_list

# Append a new message to the chat history
def append_message(chat_history, role, content):
    msg = {
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    }
    chat_history.append(msg)
