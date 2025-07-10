# Standard library imports
import os
import io

# Scientific and data processing libraries
import numpy as np
import cv2

# PDF and Image Processing
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# NLP Libraries
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate

# Web and Environment
import streamlit as st

# Custom modules
from YH.audio_transcription import transcribe_audio_files
from YH.image_captioning import image_to_text_fallback, preprocess_for_image_to_text
from YH.chat_manager import (
    init_new_conversation,
    save_chat_history,
    load_chat_history,
    list_chat_histories,
    append_message
)

os.environ["PATH"] += os.pathsep +r'C:\Release-24.08.0-0\poppler-24.08.0\Library\bin'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'







@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")


@st.cache_resource
def load_llm():
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=1024,  # Keeps the maximum length the same
        temperature=0.4,  # Keeps the temperature the same for predictability
        top_k=10,         # Reduced from 15 to make choices more focused
        top_p=0.65,       # Reduced from 0.9 to narrow down the probability distribution
        repetition_penalty=1.0,  # Keeps repetition penalty the same
        do_sample= False,  
        typical_p=0.65,   # Keeps typical p the same for now
        huggingfacehub_api_token="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",
        model_kwargs={
            
            "min_length": 200,  # Encourages longer responses
            "length_penalty": 1.2,  # Adjust if needed based on desired output length
            "no_repeat_ngram_size": 3,  # Prevents repetitive n-grams
            "early_stopping": True,  # Stops generation when a good answer is found
            
            "eos_token_id": 50256  # Specifies an end-of-sequence token if applicable
        }
    )
    return llm


def ocr_images(images, use_ocr):
    """
    Perform OCR on a list of uploaded images and return extracted text.
    
    :param images: List of image files
    :param use_ocr: Whether to use OCR for text extraction
    :return: Extracted text from images
    """
    all_text = []
    for img in images:
        try:
            # Load the image in PIL
            image = Image.open(img)
            
            # Determine text extraction method based on use_ocr flag
            if use_ocr:
                # Preprocess image to improve OCR accuracy
                preprocessed_image = preprocess_image(image)
                preprocessed_caption_image = preprocess_for_image_to_text(image)
                
                # Perform OCR
                ocr_text = pytesseract.image_to_string(preprocessed_image, config='--psm 6 -c preserve_interword_spaces=1')
                
                # If OCR fails, use image-to-text fallback
                if not ocr_text or not ocr_text.strip():
                    st.warning(f"OCR failed for {img.name}. Attempting image-to-text generation...")
                    ocr_text = image_to_text_fallback([preprocessed_caption_image], "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")
            else:
                # If OCR is disabled, use image-to-text generation as primary method
                preprocessed_caption_image = preprocess_for_image_to_text(image)
                ocr_text = image_to_text_fallback([preprocessed_caption_image], "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")
            
            # Add text if found
            if ocr_text and ocr_text.strip():
                all_text.append(ocr_text.strip())
        
        except Exception as e:
            st.error(f"Error processing image {img.name}: {e}")
    
    return "\n\n".join(all_text)




import numpy as np
import cv2
from PIL import Image

def preprocess_image(image):
    """
    Preprocess image to improve OCR accuracy
    """
    # Convert PIL Image to OpenCV format
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to preprocess the image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Convert back to PIL Image
    return Image.fromarray(threshold)




def get_pdf_text(pdf_docs):
    texts = []
    for pdf in pdf_docs:
        try:
            # Read PDF bytes
            pdf_bytes = pdf.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PdfReader(pdf_file)
            number_of_pages = len(pdf_reader.pages)

            for page_number in range(number_of_pages):
                page = pdf_reader.pages[page_number]
                
                # Extract text normally first
                page_text = page.extract_text()
                
                # If no text extracted, then convert to image and use OCR
                if not page_text or not page_text.strip():
                    # Convert PDF page to image for OCR
                    images = convert_from_bytes(
                        pdf_bytes, 
                        first_page=page_number + 1, 
                        last_page=page_number + 1, 
                        fmt='png', 
                        dpi=300
                    )
                    
                    # Perform OCR on the page image
                    for image in images:
                        # Ensure image is in RGB mode
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        # Preprocess image to improve OCR
                        preprocessed_image = preprocess_image(image)
                        
                        # Perform OCR with improved configuration
                        ocr_text = pytesseract.image_to_string(
                            preprocessed_image, 
                            config='--psm 11 -c preserve_interword_spaces=1'
                        )
                        
                        # Add OCR text if it's not empty
                        if ocr_text and ocr_text.strip():
                            page_text = ocr_text.strip()
                            break
                
                # Add the text if found
                if page_text and page_text.strip():
                    texts.append(page_text.strip())
        
        except Exception as e:
            st.error(f"Error processing PDF {pdf.name}: {e}")
    
    # Return all extracted texts joined by double newline to separate pages
    return '\n\n'.join(texts)

def tag_text_chunks(raw_text, source_type):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=10000,
        chunk_overlap=2000,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return [{"text": chunk, "source": source_type} for chunk in chunks]

def get_vectorstore(text_chunks):
    embeddings = load_embeddings()
    vectorstore = FAISS.from_texts([chunk["text"] for chunk in text_chunks], embeddings)
    return vectorstore

def build_combined_vectorstore(pdf_chunks, image_chunks, audio_chunks):
    vectorstores = []
    if pdf_chunks:
        vectorstores.append(get_vectorstore(pdf_chunks))
    if image_chunks:
        vectorstores.append(get_vectorstore(image_chunks))
    if audio_chunks:
        vectorstores.append(get_vectorstore(audio_chunks))
    
    # Handle empty case
    if not vectorstores:
        return None  
        
    if len(vectorstores) == 1:
        return vectorstores[0]
        
    combined_vectorstore = vectorstores[0]
    for vs in vectorstores[1:]:
        combined_vectorstore.merge_from(vs)
    
    return combined_vectorstore

def build_memory_from_chat_history(chat_history):
    from langchain.schema import AIMessage, HumanMessage
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    return messages

def get_conversation_chain(vectorstore, chat_history):
    llm = load_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    preloaded_messages = build_memory_from_chat_history(chat_history)
    for m in preloaded_messages:
        memory.chat_memory.add_message(m)

  
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("No conversation active. Please start/load a conversation.")
        return
    if st.session_state.vectorstore is None:
        st.warning("No context available. Please process PDFs or audio first.")
        return
    if not user_question.strip():
        st.warning("Please enter a question.")
        return
    with st.spinner("Generating response..."):
       
        response = st.session_state.conversation({'question':user_question})
    new_history = []
    for msg in response['chat_history']:
        role = 'assistant' if msg.type == 'ai' else 'user'
        append_message(new_history, role, msg.content)
    st.session_state.chat_history = new_history

    # Save updated chat history along with text chunks
    pdf_chunks = st.session_state.pdf_chunks if 'pdf_chunks' in st.session_state else []
    image_chunks = st.session_state.image_chunks if 'image_chunks' in st.session_state else []
    audio_chunks = st.session_state.audio_chunks if 'audio_chunks' in st.session_state else []
    if st.session_state.current_chat_file:
        save_chat_history(st.session_state.chat_history, st.session_state.current_chat_file, pdf_chunks=pdf_chunks, image_chunks=image_chunks, audio_chunks=audio_chunks)

def display_chat_history():
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'''
                    <div class="user-message">
                        <div class="avatar">👤</div>
                        <div class="message-content">{msg["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="bot-message">
                        <div class="avatar">🤖</div>
                        <div class="message-content">{msg["content"]}</div>
                    </div>
                ''', unsafe_allow_html=True)

def clear_current_conversation():
    st.session_state.chat_history = []
    st.session_state.conversation = None
    st.session_state.pdf_chunks = []
    st.session_state.current_chat_file = None
    st.session_state.vectorstore = None
    st.session_state.image_chunks = []
    st.session_state.audio_chunks = []


def update_vectorstore():
    """
    Updates the combined vectorstore based on the current session state.
    """
    pdf_chunks = st.session_state.get('pdf_chunks', [])
    image_chunks = st.session_state.get('image_chunks', [])
    audio_chunks = st.session_state.get('audio_chunks', [])
    st.session_state.vectorstore = build_combined_vectorstore(pdf_chunks, image_chunks, audio_chunks)

def main():
    # Session state initialization
    if 'conversation' not in st.session_state:
        st.session_state.conversation = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_chat_file' not in st.session_state:
        st.session_state.current_chat_file = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_chunks' not in st.session_state:
        st.session_state.pdf_chunks = []
    if 'image_chunks' not in st.session_state:
        st.session_state.image_chunks = []
    if 'audio_chunks' not in st.session_state:
        st.session_state.audio_chunks = []

    st.title("MistralAI Chatbot")
    st.markdown("Upload your PDFs, Images, and optionally your audio files for transcription to chat intelligently with their contents!")

    with st.sidebar:
        st.header("Conversation Management")

        histories = list_chat_histories()
        history_options = [("New Conversation", "New")] + histories
        selected_history = st.selectbox(
            "Select a conversation to continue:",
            options=[display for _, display in history_options],
            index=0
        )

        if selected_history == "New Conversation":
            if st.button("Start New Conversation"):
                filename = init_new_conversation()
                st.session_state.chat_history = []
                st.session_state.current_chat_file = filename
                st.session_state.conversation = None
                st.session_state.vectorstore = None
                st.session_state.pdf_chunks = []
                st.session_state.image_chunks = []
                st.session_state.audio_chunks = []
                st.success(f"New conversation started: {os.path.basename(filename)}")
                st.rerun()
        else:
            # Existing conversation chosen
            for fn, disp in history_options:
                if disp == selected_history:
                    selected_file = fn
                    break
            if st.button("Load Selected Conversation"):
                messages, pdf_chunks,image_chunks, audio_chunks = load_chat_history(selected_file)
                st.session_state.chat_history = messages
                st.session_state.current_chat_file = selected_file
                st.session_state.conversation = None
                st.session_state.pdf_chunks = pdf_chunks
                st.session_state.image_chunks = image_chunks
                st.session_state.audio_chunks = audio_chunks
                if pdf_chunks or image_chunks or audio_chunks:
                    try:
                        update_vectorstore()
                        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, st.session_state.chat_history)
                        st.success(f"Loaded conversation with context: {os.path.basename(selected_file)}")
                    except Exception as e:
                        st.error(f"Error rebuilding context from text chunks: {e}")
                else:
                    st.warning("No PDF, image, or audio context found in this conversation. Process documents again if needed.")
                st.success(f"Loaded conversation: {os.path.basename(selected_file)}")

        # Clear Current Conversation
        if st.button("New  Conversation"):
            st.session_state.chat_history = []
            st.session_state.conversation = None
            st.session_state.pdf_chunks = []
            st.session_state.current_chat_file = None
            st.session_state.vectorstore = None
            st.session_state.image_chunks = []
            st.session_state.audio_chunks = []
            st.success("New conversation created!")

        st.header("📤 Document Upload")
        tab_pdf, tab_images, tab_audio = st.tabs(["PDF Upload", "Image Upload", "Audio Upload"])

        with tab_pdf:
            st.header("PDF Upload")
            pdf_docs = st.file_uploader(
                "Choose PDF files",
                type=['pdf'],
                accept_multiple_files=True,
                help="Upload one or more PDF documents"
             )

            if st.button("Process PDFs"):
                if pdf_docs:
                    with st.spinner("Processing PDFs..."):
                        raw_text = get_pdf_text(pdf_docs)
                    if raw_text.strip() == "":
                        st.error("No text could be extracted from the uploaded PDFs.")
                    else:
                        pdf_chunks = tag_text_chunks(raw_text, "pdf")
                        st.session_state.pdf_chunks.extend(pdf_chunks)
                        update_vectorstore()

                        if st.session_state.current_chat_file is None:
                            # No conversation chosen, start a new one
                            filename = init_new_conversation(pdf_chunks=st.session_state.pdf_chunks)
                            st.session_state.current_chat_file = filename
                            st.session_state.chat_history = []
                            st.success("No conversation was selected, so a new one was started.")
                        else:
                            # Update existing conversation with these pdf_chunks
                            save_chat_history(st.session_state.chat_history, st.session_state.current_chat_file, pdf_chunks=st.session_state.pdf_chunks, image_chunks=st.session_state.image_chunks, audio_chunks=st.session_state.audio_chunks)

                        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, st.session_state.chat_history)
                        st.success("PDFs processed and conversation chain updated!")
                else:
                    st.warning("Please upload at least one PDF file.")

    
        with tab_images:
            st.header("🖼 Image Upload")
        
        # Add OCR toggle for images
            use_image_ocr = st.checkbox(
                "Enable OCR for Images", 
                value=False,  # Default to OCR enabled
                help="When checked, attempts to extract text from images using OCR or image-to-text generation. Uncheck to use only image-to-text generation."
             )

            image_files = st.file_uploader(
                "Choose Image files (PNG/JPG/JPEG)",
                type=['png', 'jpg', 'jpeg'],
                accept_multiple_files=True,
                help="Upload one or more image files"
            )

            if st.button("Process Images"):
                if image_files:
                    with st.spinner("Processing images..."):
                    # Pass the OCR toggle to the ocr_images function
                        ocr_text = ocr_images(image_files, use_ocr=use_image_ocr)
                    
                    if ocr_text.strip() == "":
                        st.warning("No text could be extracted from the uploaded images.")
                    else:
                        image_chunks = tag_text_chunks(ocr_text, "image")
                        st.session_state.image_chunks.extend(image_chunks)
                        update_vectorstore()

                        if st.session_state.current_chat_file is None:
                            # No conversation chosen, start a new one
                            filename = init_new_conversation(image_chunks=st.session_state.image_chunks)
                            st.session_state.current_chat_file = filename
                            st.session_state.chat_history = []
                            st.success("No conversation was selected, so a new one was started.")
                        else:
                            # Update existing conversation
                            save_chat_history(st.session_state.chat_history, st.session_state.current_chat_file, pdf_chunks=st.session_state.pdf_chunks, image_chunks=st.session_state.image_chunks, audio_chunks=st.session_state.audio_chunks)

                        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, st.session_state.chat_history)
                        st.success("Images processed and conversation chain updated!")
                else:
                    st.warning("Please upload at least one image file.")

        with tab_audio:
            st.header("🎤 Audio Upload")
            audio_files = st.file_uploader(
            "Choose Audio files (MP3/WAV)",
            type=['mp3','wav'],
            accept_multiple_files=True,
            help="Upload one or more audio files"
            )

            if st.button("Process Audio"):
                if audio_files:
                    with st.spinner("Transcribing Audio..."):
                    # Transcribe audio via HuggingFace ASR pipeline from audio_transcription.py
                        transcription_text = transcribe_audio_files(audio_files, model_name="openai/whisper-base")
                    if transcription_text.strip() == "":
                        st.warning("No transcription could be extracted from the uploaded audio.")
                    else:
                        audio_chunks = tag_text_chunks(transcription_text, "audio")
                        st.session_state.audio_chunks.extend(audio_chunks)
                        update_vectorstore()

                        if st.session_state.current_chat_file is None:
                            # No conversation chosen, start a new one with combined chunks
                            filename = init_new_conversation(audio_chunks=st.session_state.audio_chunks)
                            st.session_state.current_chat_file = filename
                            st.session_state.chat_history = []
                            st.success("No conversation was selected, so a new one was started.")
                        else:
                            # Update existing conversation
                            save_chat_history(st.session_state.chat_history, st.session_state.current_chat_file, pdf_chunks=st.session_state.pdf_chunks, image_chunks=st.session_state.image_chunks,audio_chunks=st.session_state.audio_chunks)

                        st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore, st.session_state.chat_history)
                        st.success("Audio processed and conversation chain updated!")
                else:
                    st.warning("Please upload at least one audio file.")

        

    chat_container = st.container()
    with chat_container:
        display_chat_history()
    

    user_question = st.chat_input(
    "Type your Message here"
    )

    if  user_question:
        if st.session_state.conversation is None:
            st.warning("No conversation active. Please start/load a conversation.")
        elif st.session_state.vectorstore is None:
            st.warning("No context available. Please process PDFs, images, or audio files first.")
        else:
            handle_userinput(user_question)
            st.rerun()

   
if __name__ == "__main__":
    main()