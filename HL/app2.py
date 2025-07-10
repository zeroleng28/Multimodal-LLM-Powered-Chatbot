import streamlit as st

import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from PIL import Image
from HL.audio_to_text import process_audio_files, convert_mp3_to_wav
import pytesseract
from pdf2image import convert_from_bytes
from pathlib import Path
import io
from datetime import datetime

from HL.chat_manager1 import (
    init_new_conversation,
    save_chat_history,
    load_chat_history,
    list_chat_histories,
    append_message
)


# Configure Tesseract path with better error handling
def setup_tesseract():
    """Setup and verify Tesseract installation"""
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    try:
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            # Verify installation
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract version: {version}")
            return True
        else:
            print(f"Tesseract not found at: {tesseract_path}")
            return False
    except Exception as e:
        print(f"Error setting up Tesseract: {str(e)}")
        return False

# Gemini Configuration
GEMINI_API_KEY = "AIzaSyAtKpVYlM7mAZVRGn3hlxQXkKA4HLlMWnA"
GEMINI_MODEL = "gemini-1.5-flash"



@st.cache_resource
def load_llm():
    """
    Initialize and configure Gemini 1.5 Flash model
    """
    try:
        # Configure Gemini with API key
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Define generation config
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,          # Controls randomness (0.0 - 1.0)
            top_p=0.8,               # Nucleus sampling parameter
            top_k=40,                # Top-k sampling parameter
            max_output_tokens=2048,   # Maximum length of response
            candidate_count=1,        # Number of generated responses
        )
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config
        )
        
        # Start a chat session
        chat = model.start_chat(history=[]) 
        
        st.success("Gemini model initialized successfully!")
        return chat

    except Exception as e:
        error_message = f"Error initializing Gemini model: {str(e)}"
        st.error(error_message)
        print(f"Detailed error: {type(e).__name__} - {str(e)}")
        return None

# Usage in your main code
if 'app2_llm' not in st.session_state:
    st.session_state.app2_llm = load_llm()

# Safety check before using the model
def ensure_model_loaded():
    if st.session_state.app2_llm is None:
        st.error("Model not properly initialized. Please check your API key and try again.")
        return False
    return True


def get_pdf_text(pdf_docs):
    """Extract text from PDF files"""
    texts = []
    for pdf in pdf_docs:
        try:
            # Debug print
            print(f"Processing PDF: {pdf.name}")
            
            pdf_bytes = pdf.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_file)
            
            # Debug print
            print(f"Number of pages: {len(pdf_reader.pages)}")
            
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                text = page.extract_text()
                
                if text.strip():
                    texts.append(text)
                    print(f"Extracted text from page {page_number + 1}")
                else:
                    print(f"No text found on page {page_number + 1}, trying OCR")
                    # OCR processing
                    images = convert_from_bytes(
                        pdf_bytes,
                        first_page=page_number + 1,
                        last_page=page_number + 1
                    )
                    for image in images:
                        text = pytesseract.image_to_string(image)
                        if text.strip():
                            texts.append(text)
                            print(f"OCR extracted text from page {page_number + 1}")
        
        except Exception as e:
            print(f"Error processing PDF {pdf.name}: {type(e).__name__} - {str(e)}")
            st.error(f"Error processing PDF {pdf.name}: {str(e)}")
    
    return "\n".join(texts)

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def get_image_text(image_files):
    """
    Extract text from uploaded image files.
    Args:
        image_files (list): List of uploaded image files.
    Returns:
        str: Extracted text from all images
    """
    texts = []
    for image_file in image_files:
        try:
            # Show processing status
            st.info(f"Processing {image_file.name}...")
            
            # Read image
            image = Image.open(image_file)
            st.image(image, caption=f"Processing: {image_file.name}", width=300)
            
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Add debug information
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
            
            try:
                # Extract text using OCR with improved configuration
                text = pytesseract.image_to_string(
                    image,
                    lang='eng',  # Language
                    config='--psm 3 --oem 3'  # Page segmentation mode and OCR Engine mode
                )
                
                if text.strip():
                    texts.append(text)
                    print(f"Successfully extracted text from: {image_file.name}")
                    print(f"Extracted text: {text[:100]}...")  # Print first 100 chars
                else:
                    print(f"No text found in image: {image_file.name}")
                    
            except Exception as ocr_error:
                print(f"OCR Error: {str(ocr_error)}")
                st.error(f"OCR Error for {image_file.name}: {str(ocr_error)}")
                
        except Exception as e:
            print(f"Error processing image {image_file.name}: {type(e).__name__} - {str(e)}")
            st.error(f"Error processing image {image_file.name}: {str(e)}")
    
    combined_text = "\n".join(texts)
    
    # Debug print
    if combined_text.strip():
        print("Successfully extracted text from images")
        print(f"Total text length: {len(combined_text)}")
    else:
        print("No text was extracted from any images")
    
    return combined_text

# Add a function to verify Tesseract installation
def verify_tesseract():
    """Verify Tesseract OCR installation"""
    try:
        version = pytesseract.get_tesseract_version()
        st.success(f"Tesseract OCR version {version} is installed")
        return True
    except Exception as e:
        st.error("""
        Tesseract is not properly installed. Please follow these steps:
        
        1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
        2. Install it (recommended path: C:\\Program Files\\Tesseract-OCR)
        3. Add the installation path to your system PATH
        4. Restart your computer
        
        Error: {str(e)}
        """)
        return False

def get_gemini_response(question, context=""):
    """
    Get response from the loaded Gemini model
    
    Args:
        question (str): User's question
        context (str): Optional context from PDF
    
    Returns:
        Generator of response chunks or error message
    """
    if not ensure_model_loaded():
        return "Model not available. Please try again later."
        
    try:
        # Format prompt with context if available
        prompt = f"""
        Instructions: You are a helpful AI assistant. 
        {f'Use the following context to answer questions: {context}' if context else 'Provide a helpful response.'}
        
        Question: {question}
        """
        
        # Get streaming response
        response = st.session_state.app2_llm.send_message(
            prompt,
            stream=True
        )
        
        return response
        
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        st.error(error_message)
        return error_message


def handle_userinput(user_question):
    if not user_question.strip():
        st.warning("Please enter a question.")
        return

    if 'app2_llm' not in st.session_state:
        st.warning("LLM model not initialized.")
        return

    # Initialize context as empty
    context = ""

    # Check for each type of content and prepare the context
    if 'pdf_content' in st.session_state and st.session_state.pdf_content:
        context += "PDF Content: " + st.session_state.pdf_content + "\n"
    
    if 'image_content' in st.session_state and st.session_state.image_content:
        context += "Image Description: " + st.session_state.image_content + "\n"

    if 'audio_content' in st.session_state and st.session_state.audio_content:
        context += "Audio Transcription: " + st.session_state.audio_content

    if not context:
        context = "No additional context available."

    with st.spinner("Generating response..."):
        # Assuming get_gemini_response is a function that takes a question and context
        response_stream = get_gemini_response(user_question, context)
        response_text = ""
        response_area = st.empty()

        try:
            for chunk in response_stream:
                response_text += chunk.text
                response_area.markdown(response_text)
        except Exception as e:
            if isinstance(response_stream, str):  # If it's an error message
                response_text = response_stream
            else:
                response_text = f"Error generating response: {str(e)}"

        # Update chat history
        append_message(st.session_state.app2_chat_history, "User", user_question)
        append_message(st.session_state.app2_chat_history, "Gemini", response_text)

        # Save chat history along with the context
        if st.session_state.app2_current_chat_file:
            save_chat_history(
                st.session_state.app2_chat_history,
                st.session_state.app2_current_chat_file,
                pdf_content=st.session_state.pdf_content,
                image_content=st.session_state.image_content,
                audio_content=st.session_state.audio_content
            )


def reset_chat():
    st.session_state.app2_chat_history = []
    st.session_state.pdf_content = ""
    st.session_state.image_content = ""
    



def display_chat_history():
    """Display chat history with improved UI"""
    for msg in st.session_state.app2_chat_history:
        if msg["role"] == "User":
            # User messages on the right with black text
            st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
                    <div style='background-color: #DCF8C6; padding: 10px; color: black; 
                        border-radius: 15px; max-width: 70%; margin-left: 20%;'>
                        <strong>You:</strong><br>{msg["content"]}</br>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        elif msg["role"] == "Gemini":
            # Assistant messages on the left with black text
            if "<" in msg["content"] or ">" in msg["content"]:
                # Simple escaping if there's a risk of breaking HTML
                safe_content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
            else:
                safe_content = msg["content"]

            st.markdown(f"""
                <div style='display: flex; justify-content: flex-start; margin-bottom: 10px;'>
                    <div style='background-color: #E8E8E8; padding: 10px; color: black; 
                                border-radius: 15px; max-width: 70%; margin-right: 20%;'>
                        <strong>Assistant:</strong><br>{safe_content}</br>
                    </div>
                </div>
            """, unsafe_allow_html=True)


def main():
    # Initialize session states
    if 'app2_llm' not in st.session_state:
        st.session_state.app2_llm = load_llm()
    if 'app2_chat_history' not in st.session_state:
        st.session_state.app2_chat_history = []
    if 'app2_current_chat_file' not in st.session_state:
        st.session_state.app2_current_chat_file = None
    if 'pdf_content' not in st.session_state:
        st.session_state.pdf_content = ""
    if 'image_content' not in st.session_state:
        st.session_state.image_content = ""
    if 'audio_content' not in st.session_state:
        st.session_state.audio_content =""

    st.title("Gemini Chatbot PDF & Image Chat Companion")
    st.markdown("Upload your PDFs or Image and chat intelligently with their contents!")

    with st.sidebar:
        st.header("Conversation Management")

        histories = list_chat_histories()
        history_options = [("New Conversation", "new")] + histories
        selected_history = st.selectbox(
            "Select a conversation to continue:",
            options=[display for _, display in history_options],
            index=0
        )

        if selected_history == "New Conversation":
            if st.button("Start New Conversation"):
                filename = init_new_conversation()
                st.session_state.app2_chat_history = []
                st.session_state.app2_current_chat_file = filename
                st.session_state.pdf_content = ""
                st.session_state.image_content = ""
                st.session_state.audio_content =""
                st.session_state.app2_llm = load_llm()
                st.success(f"New conversation started: {os.path.basename(filename)}")
                st.rerun()
        else:
            for fn, disp in history_options:
                if disp == selected_history:
                    selected_file = fn
                    break
            if st.button("Load Selected Conversation"):
                messages, pdf_content,image_content, audio_content = load_chat_history(selected_file)
                st.session_state.app2_chat_history = messages
                st.session_state.app2_current_chat_file = selected_file
                st.session_state.image_content = image_content
                st.session_state.pdf_content = pdf_content
                st.session_state.audio_content = audio_content
                st.session_state.app2_llm = load_llm()
                st.success(f"Loaded conversation: {os.path.basename(selected_file)}")

        if st.button("Clear Current Conversation"):
            st.session_state.app2_chat_history = []
            st.session_state.app2_current_chat_file = None
            st.session_state.pdf_content = ""
            st.session_state.image_content = ""
            st.session_state.audio_content =""
            st.session_state.app2_llm = load_llm()
            st.success("Current conversation cleared!")

        st.header("📤 Document Upload")
        
        # Add audio tab
        tab1, tab2, tab3 = st.tabs(["PDF Upload", "Image Upload", "Audio Upload"])
        
        with tab1:
            pdf_docs = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF documents"
            )

            if st.button("Process PDFs", key="process_pdf_btn"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        pdf_text = get_pdf_text(pdf_docs)
                        if pdf_text.strip() == "":
                            st.error("No text could be extracted from the uploaded PDFs.")
                        else:
                            st.session_state.pdf_content = pdf_text
                            
                            if st.session_state.app2_current_chat_file is None:
                                # Pass both pdf_content and image_content
                                filename = init_new_conversation(
                                    pdf_content=pdf_text,
                                    image_content="",
                                    audio_content=""
                                )
                                st.session_state.app2_current_chat_file = filename
                                st.session_state.app2_chat_history = []
                                st.success("New conversation started with PDF content.")
                            else:
                                save_chat_history(
                                    st.session_state.app2_chat_history,
                                    st.session_state.app2_current_chat_file,
                                    pdf_content=pdf_text,
                                    image_content="",
                                    audio_content=""
                                )
                            st.success("PDFs processed successfully!")
                else:
                    st.warning("Please upload at least one PDF file.")
        
        with tab2:
            # Add Tesseract status check
            if st.checkbox("Check Tesseract Status"):
                if setup_tesseract():
                    st.success("✅ Tesseract is ready!")
                else:
                    st.error("""
                    Tesseract is not properly configured. Please:
                    1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
                    2. Install to: C:\\Program Files\\Tesseract-OCR
                    3. Add to PATH
                    4. Restart your computer
                    """)
                    return

            image_files = st.file_uploader(
                "Choose Image files",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload one or more image files"
            )

            if st.button("Process Images", key="process_img_btn"):
                if image_files:
                    with st.spinner("Processing Images..."):
                        image_text = get_image_text(image_files)
                        if image_text:
                            st.session_state.image_content = image_text
                            
                            # Show extracted text
                            with st.expander("View Extracted Text"):
                                st.write(image_text)
                            
                            if st.session_state.app2_current_chat_file is None:
                                filename = init_new_conversation(
                                    pdf_content="",
                                    image_content=image_text,
                                    audio_content=""
                                )
                                st.session_state.app2_current_chat_file = filename
                                st.session_state.app2_chat_history = []
                            else:
                                save_chat_history(
                                    st.session_state.app2_chat_history,
                                    st.session_state.app2_current_chat_file,
                                    audio_content="",
                                    pdf_content="",
                                    image_content=image_text
                                )
                            st.success("Images processed successfully!")
                        else:
                            st.warning("No text could be extracted from the images")
                else:
                    st.warning("Please upload at least one image file")
        
        with tab3:
            audio_files = st.file_uploader(
                "Choose Audio files",
                type=['wav', 'mp3'],
                accept_multiple_files=True,
                help="Upload one or more audio files (WAV or MP3)"
            )

            if st.button("Process Audio", key="process_audio_btn"):
                if audio_files:
                    
                    audio_text = process_audio_files(audio_files)
                    if audio_text:
                        st.session_state.audio_content = audio_text
                        
                        if st.session_state.app2_current_chat_file is None:
                            filename = init_new_conversation(
                                pdf_content="",
                                image_content="",
                                audio_content=audio_text
                            )
                            st.session_state.app2_current_chat_file = filename
                            st.session_state.app2_chat_history = []
                        else:
                            save_chat_history(
                                st.session_state.app2_chat_history,
                                st.session_state.app2_current_chat_file,
                                pdf_content="",
                                image_content="",
                                audio_content=audio_text
                            )
                else:
                    st.warning("Please upload at least one audio file")


    st.header("💬 Chat with Your Documents")
    display_chat_history()

    
    user_input = st.chat_input("Type your message here")
       
        
    if  user_input:
        handle_userinput(user_input)
        st.rerun()

    

if __name__ == "__main__":
    main()
