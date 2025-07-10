# chat_manager.py
import json
import os
from datetime import datetime
import streamlit as st



CHAT_HISTORY_DIR = "E:/Desktop/NLP_ASG/chat_histories_HL"

def ensure_directory_exists():
    """Create chat histories directory if it doesn't exist"""
    try:
        if not os.path.exists(CHAT_HISTORY_DIR):
            os.makedirs(CHAT_HISTORY_DIR)
            print(f"Created directory: {CHAT_HISTORY_DIR}")
    except Exception as e:
        print(f"Error creating directory: {str(e)}")

def generate_chat_filename():
    """Generate a unique filename for the chat history"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"chat_{timestamp}.json"
    return os.path.join(CHAT_HISTORY_DIR, filename)

def init_new_conversation(pdf_content="", image_content="", audio_content=""):
    """Initialize a new conversation with optional content"""
    try:
        ensure_directory_exists()
        filename = generate_chat_filename()
        
        # Create initial data structure
        data = {
            "messages": [],
            "pdf_content": pdf_content,
            "image_content": image_content,
            "audio_content": audio_content
        }
        
        # Save with proper formatting
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Created new conversation file: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error creating conversation: {str(e)}")
        st.error(f"Failed to create new conversation: {str(e)}")
        return None

def save_chat_history(chat_history, filename=None, pdf_content="", image_content="", audio_content=""):
    """Save chat history and content to file"""
    try:
        # Ensure the directory exists
        ensure_directory_exists()
        
        # Generate new filename if none provided
        if filename is None:
            filename = generate_chat_filename()
        
        data = {
            "messages": chat_history,
            "pdf_content": pdf_content,
            "image_content": image_content,
            "audio_content": audio_content,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save with proper formatting
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved chat history to: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error saving chat history: {str(e)}")
        st.error(f"Failed to save chat history: {str(e)}")
        return None

def load_chat_history(filename):
    """Load chat history and content from file"""
    print(f"Loading chat history from: {filename}")
    
    try:
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return [], "", "", ""
        
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read().strip()
            
            # Handle empty file
            if not content:
                print(f"Empty file: {filename}")
                return [], "", "", ""
            
            try:
                # Parse JSON content
                data = json.loads(content)
                messages = data.get("messages", [])
                pdf_content = data.get("pdf_content", "")
                image_content = data.get("image_content", "")
                audio_content = data.get("audio_content", "")
                
                print(f"Successfully loaded chat history from: {filename}")
                return messages, pdf_content, image_content, audio_content
                
            except json.JSONDecodeError as je:
                print(f"JSON decode error in {filename}: {str(je)}")
                return [], "", "", ""
                
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        st.error(f"Failed to load chat history: {str(e)}")
        return [], "", "", ""

def list_chat_histories():
    """List all available chat histories"""
    try:
        ensure_directory_exists()
        files = [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith(".json")]
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(CHAT_HISTORY_DIR, x)), reverse=True)
        
        history_list = []
        for f in files:
            try:
                # Read the file to get the timestamp
                with open(os.path.join(CHAT_HISTORY_DIR, f), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    timestamp = data.get('timestamp', '')
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp)
                        display_name = dt.strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        display_name = f.replace('.json', '')
                    
                history_list.append((os.path.join(CHAT_HISTORY_DIR, f), display_name))
            except Exception:
                # Fallback to filename if there's an error
                history_list.append((os.path.join(CHAT_HISTORY_DIR, f), f))
        
        return history_list
        
    except Exception as e:
        print(f"Error listing chat histories: {str(e)}")
        st.error(f"Failed to list chat histories: {str(e)}")
        return []

def append_message(chat_history, role, content):
    """Append a new message to the chat history"""
    try:
        if chat_history is None:
            chat_history = []
            
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        chat_history.append(msg)
        
    except Exception as e:
        print(f"Error appending message: {str(e)}")
        st.error(f"Failed to append message: {str(e)}")

# Initialize chat directory when module is loaded
ensure_directory_exists()
