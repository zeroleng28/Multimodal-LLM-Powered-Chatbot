import streamlit as st
import os
import json
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import Runnable
from langchain.schema.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from PIL import Image
import io
from transformers import pipeline
import easyocr
import numpy as np
import base64
import speech_recognition as sr
from tempfile import NamedTemporaryFile
from typing import Dict, List
from pydub import AudioSegment
import requests

AudioSegment.converter = r"C:/ffmpeg/ffmpeg.exe"


# Debug flag
DEBUG = True

# Configuration
CONFIG = {
    "huggingface_api_key": "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",
    "model_repo_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "api_url": "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct",
    "model_config": {
        "max_new_tokens": 1024,
        "temperature": 0.2,
        "repetition_penalty": 1.2,
        "context_length": 4096,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50,
        "model_kwargs": {  # Additional model-specific options
            "min_length": 50,
            "length_penalty": 2.0,
            "no_repeat_ngram_size": 5,
            "early_stopping": True,
            "eos_token_id": 50256,
            "pad_token_id": 0
        }
    },
    "chat_history_path": "E:/Desktop/NLP_ASG/chat_histories_ln"
}

# Utility Functions
def debug_log(message):
    if DEBUG:
        print(message)

def save_chat_history_json(chat_history, file_path):
    if not file_path.endswith('.json'):
        file_path += '.json'
    
    with open(file_path, "w") as f:
        json_data = []
        for message in chat_history.messages:
            message_dict = {
                "type": message.type,
                "content": message.content,
                "additional_kwargs": message.additional_kwargs,
                "example": message.example
            }
            json_data.append(message_dict)
        json.dump(json_data, f)

def load_chat_history(file_path):
    try:
        with open(file_path, "r") as f:
            json_data = json.load(f)
            messages = []
            for message in json_data:
                if message["type"] == "human":
                    messages.append(HumanMessage(
                        content=message["content"],
                        additional_kwargs=message.get("additional_kwargs", {}),
                        example=message.get("example", False)
                    ))
                elif message["type"] == "ai":
                    messages.append(AIMessage(
                        content=message["content"],
                        additional_kwargs=message.get("additional_kwargs", {}),
                        example=message.get("example", False)
                    ))
            return messages
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading chat history: {e}")
        return []

# LLM Chain Implementation
def create_llm():
    try:
        api_url = CONFIG['api_url']
        headers = {
            "Authorization": f"Bearer {CONFIG['huggingface_api_key']}"
        }
        return {
            "api_url": api_url,
            "headers": headers,
            "parameters": CONFIG['model_config']
        }
    except Exception as e:
        raise RuntimeError(f"Error creating LLM with API URL: {e}")

def send_request_to_huggingface(prompt):
    llm_config = create_llm()
    api_url = llm_config["api_url"]
    headers = llm_config["headers"]
    
    payload = {
        "inputs": prompt,
        "parameters": llm_config["parameters"],
        "options": {
            "wait_for_model": True,
            "use_cache": False
        }
    }

    try:
        debug_log(f"Sending request to: {api_url}")
        debug_log(f"Payload: {payload}")
        
        response = requests.post(api_url, headers=headers, json=payload)
        debug_log(f"Response status code: {response.status_code}")
        debug_log(f"Response content: {response.text}")
        
        response.raise_for_status()
        output = response.json()
        return output[0]["generated_text"] if isinstance(output, list) else output.get("generated_text", "")
    except requests.exceptions.RequestException as e:
        error_detail = f"API Request failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            error_detail += f"\nResponse: {e.response.text}"
        debug_log(error_detail)
        raise RuntimeError(error_detail)
    except Exception as e:
        error_detail = f"Unexpected error: {str(e)}"
        debug_log(error_detail)
        raise RuntimeError(error_detail)

class HuggingFaceRunnable(Runnable):
    def invoke(self, input, config=None, **kwargs):
        debug_log(f"Input received in HuggingFaceRunnable: {input}")
        
        # Handle StringPromptValue
        from langchain_core.prompt_values import StringPromptValue
        if isinstance(input, StringPromptValue):
            return send_request_to_huggingface(input.text)
        
        if isinstance(input, str):
            return send_request_to_huggingface(input)
        elif isinstance(input, dict):
            # Combine PDF and image contexts if available
            prompt = ""
            if "context" in input and "question" in input:
                prompt = f"Context from PDF: {input['context']}\n"
            if "image_context" in input:
                prompt += f"Image Context: {input['image_context']}\n"
            
            # Add the question
            question = input.get("question") or input.get("input_text")
            if question:
                prompt += f"Question: {question}"
            
            return send_request_to_huggingface(prompt)
        
        raise ValueError(f"Input must be a string or a dict with appropriate keys. Received: {type(input)}, {input}")

def load_normal_chain(chat_history):
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True
    )
    return HuggingFaceRunnable()

# Streamlit App Functions
def save_chat_history(chat_history, force_save=False):
    if chat_history.messages:
        if "chat_filename" not in st.session_state and st.session_state.session_key == "new_session":
            st.session_state.chat_filename = datetime.now().strftime("Chat_%Y%m%d_%H%M%S")
        
        filename = st.session_state.chat_filename
        if not filename.endswith('.json'):
            filename += '.json'
        
        file_path = os.path.join(CONFIG["chat_history_path"], filename)
        save_chat_history_json(chat_history, file_path)
        debug_log(f"Chat history saved to: {file_path}")
        
        if st.session_state.session_key == "new_session" and not force_save:
            st.rerun()

def initialize_session():
    chat_history = StreamlitChatMessageHistory(key="chat_messages")
    
    if st.session_state.session_key == "new_session":
        chat_history.clear()
        st.session_state.messages = []
        debug_log("New session initialized")
    else:
        try:
            filename = st.session_state.session_key
            if not filename.endswith('.json'):
                filename += '.json'
            
            file_path = os.path.join(CONFIG["chat_history_path"], filename)
            messages = load_chat_history(file_path)
            
            chat_history.clear()
            st.session_state.messages = []
            
            for message in messages:
                chat_history.add_message(message)
                st.session_state.messages.append({
                    "type": message.type,
                    "content": message.content
                })
            
            st.session_state.chat_filename = st.session_state.session_key
            debug_log(f"Chat session loaded from: {file_path}")
        except Exception as e:
            st.error(f"Error loading chat history: {e}")
            debug_log(f"Error loading chat history: {e}")
    
    return chat_history

def display_messages():
    if "messages" in st.session_state and st.session_state.messages:
        seen = set()
        unique_messages = []
        for message in st.session_state.messages:
            msg_id = (message["type"], message["content"])
            if msg_id not in seen:
                seen.add(msg_id)
                unique_messages.append(message)
        
        st.session_state.messages = unique_messages
        
        for message in unique_messages:
            with st.chat_message(message["type"]):
                st.write(message["content"])
    else:
        st.write("No messages yet. Start the conversation!")
        debug_log("No messages to display")

def transcribe_audio_file(audio_file):
    """
    Transcribes audio files (MP3 or WAV) to text using the speech_recognition library.
    """
    recognizer = sr.Recognizer()
    try:
        # Convert the uploaded audio to WAV format if necessary
        with NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            audio = AudioSegment.from_file(audio_file)
            audio.export(temp_wav_file.name, format="wav")
            temp_wav_path = temp_wav_file.name

        # Use the WAV file for transcription
        with sr.AudioFile(temp_wav_path) as source:
            audio_data = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_data)
            return transcription
    except Exception as e:
        debug_log(f"Error transcribing audio: {e}")
        return f"Error processing audio: {e}"
    finally:
        # Cleanup temporary file
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def convert_mp3_to_wav(mp3_file_path, wav_file_path):
    try:
        audio = AudioSegment.from_mp3(mp3_file_path)
        audio.export(wav_file_path, format="wav")
        return wav_file_path
    except Exception as e:
        raise RuntimeError(f"Error converting MP3 to WAV: {str(e)}")

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    if not text.strip():
        st.error("No text could be extracted from the PDF. Please try another file.")
        return None

    debug_log(f"Extracted text: {text[:500]}")  # Log first 500 characters of the text
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    debug_log(f"Chunks created: {chunks[:3]}")  # Log first 3 chunks
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    debug_log(f"Vectorstore created with {len(chunks)} chunks.")
    
    return vectorstore

def load_pdf_chain(chat_history, vectorstore):
    memory = ConversationBufferMemory(
        chat_memory=chat_history,
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

    CONFIG["model_config"].update({
        "max_new_tokens": 512,
        "temperature": 0.2,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "top_p": 0.9,
        "top_k": 50
    })
    llm = HuggingFaceRunnable()
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_kwargs={
                "k": 4,  # Increased to get more complete context
                "fetch_k": 6
            }
        ),
        memory=memory,
        return_source_documents=True
    )
    return chain

def process_image(image_file):
    """
    Process an image file by converting it to base64 first, then analyze its content.
    Returns the image analysis and extracted text.
    """
    try:
        # Convert uploaded file to base64 first
        image = Image.open(image_file)
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to base64
        buffered = io.BytesIO()
        image.save(buffered, format=image.format or 'PNG')
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Extract text using EasyOCR
        try:
            reader = easyocr.Reader(['en'])
            image_np = np.array(image)
            ocr_results = reader.readtext(image_np)
            extracted_text = '\n'.join([text[1] for text in ocr_results])
        except Exception as e:
            debug_log(f"OCR Error: {str(e)}")
            extracted_text = ""

        # Initialize image captioning pipeline
        image_to_text = pipeline("image-to-text", 
                               model="Salesforce/blip-image-captioning-large",
                               token=CONFIG["huggingface_api_key"])
        
        # Generate detailed image description
        image_description = image_to_text(image)[0]['generated_text']
        
        # Initialize image classification
        classifier = pipeline("image-classification", 
                           model="microsoft/resnet-50",
                           token=CONFIG["huggingface_api_key"])
        
        # Get classification results
        classification_results = classifier(image)
        
        result = {
            "extracted_text": extracted_text.strip(),
            "image_description": image_description,
            "classifications": classification_results[:3],
            "image_size": image.size,
            "image_format": image.format,
            "image_base64": img_base64  # Store base64 version for later use
        }
        
        debug_log(f"Image processing result: {result}")
        return result
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        debug_log(error_msg)
        raise RuntimeError(error_msg)

def get_combined_context(prompt, pdf_chain=None, image_context=None, audio_context=None):
    """Helper function to combine all document, image, and audio contexts."""
    combined_context = {"question": prompt}
    
    # Add context from all processed documents
    if "documents" in st.session_state:
        docs_context = []
        for doc_name, doc_info in st.session_state.documents.items():
            vectorstore = doc_info['vectorstore']
            relevant_docs = vectorstore.similarity_search(prompt, k=2)
            docs_context.extend([doc.page_content for doc in relevant_docs])
        if docs_context:
            combined_context["context"] = "\n".join(docs_context)
    
    # Add context from all processed images
    if "images" in st.session_state:
        img_contexts = []
        for img_name, img_info in st.session_state.images.items():
            img_contexts.append(img_info['context']['image_description'])
            if img_info['context']['extracted_text']:
                img_contexts.append(img_info['context']['extracted_text'])
        if img_contexts:
            combined_context["image_context"] = "\n".join(img_contexts)
    
    # Add audio transcription context
    if "audio_transcriptions" in st.session_state and st.session_state.audio_transcriptions:
        combined_context["audio_context"] = "\n".join(st.session_state.audio_transcriptions)

    return combined_context

def initialize_chat_chain(chat_history, documents=None, images=None, audio_transcripts=None):
    """Initialize the chat chain based on available contexts"""
    if documents or images or audio_transcripts:
        # Create a combined context chain that can handle multiple types of input
        memory = ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            memory_key="chat_history",
            output_key="answer"
        )
        
        prompt_template = PromptTemplate(
            input_variables=["context", "image_context", "audio_context", "question"],
            template="""Answer the question based on the context provided. Be concise and clear.

            Examples:
            - Q: What is AI? 
              A: AI stands for artificial intelligence, which simulates human intelligence in machines.
            - Q: 1+1=? 
              A: 1+1=2.
            - Q: How does photosynthesis work? 
              A: Photosynthesis converts sunlight into energy, producing oxygen and glucose.
            
            Available Context:
            {context}
            {image_context}
            {audio_context}
            
            Question: {question}
            
            Response:"""
        )        
        llm = HuggingFaceRunnable()
        
        # Customize retriever configuration
        retriever_config = {
            "search_kwargs": {
                "k": 4,
                "fetch_k": 6
            }
        }
        
        # Create retrievers for each document if available
        retrievers = []
        if documents:
            for doc_info in documents.values():
                if 'vectorstore' in doc_info:
                    retrievers.append(doc_info['vectorstore'].as_retriever(**retriever_config))
        
        # Create a custom chain that combines all sources
        chain = MultiSourceConversationalChain(
            llm=llm,
            retrievers=retrievers,
            memory=memory,
            prompt=prompt_template,
            images=images,
            audio_transcripts=audio_transcripts
        )
        
        return chain
    else:
        # Fall back to normal chain if no special contexts are available
        return load_normal_chain(chat_history)

class MultiSourceConversationalChain:
    def __init__(self, llm, retrievers, memory, prompt, images=None, audio_transcripts=None):
        self.llm = llm
        self.retrievers = retrievers
        self.memory = memory
        self.prompt = prompt
        self.images = images or {}
        self.audio_transcripts = audio_transcripts or []
    
    def invoke(self, input_dict):
        # Gather context from all documents
        doc_contexts = []
        if self.retrievers:
            for retriever in self.retrievers:
                relevant_docs = retriever.get_relevant_documents(input_dict["question"])
                doc_contexts.extend([doc.page_content for doc in relevant_docs])
        
        # Gather context from images
        img_contexts = []
        for img_info in self.images.values():
            if 'context' in img_info:
                img_contexts.append(img_info['context']['image_description'])
                if img_info['context']['extracted_text']:
                    img_contexts.append(img_info['context']['extracted_text'])
        
        # Prepare the combined input
        combined_input = {
            "context": "\n".join(doc_contexts) if doc_contexts else "No document context available.",
            "image_context": "\n".join(img_contexts) if img_contexts else "No image context available.",
            "audio_context": "\n".join(self.audio_transcripts) if self.audio_transcripts else "No audio context available.",
            "question": input_dict["question"]
        }
        
        # Format the prompt and get response
        formatted_prompt = self.prompt.format(**combined_input)
        response = self.llm.invoke(formatted_prompt)
        
        return response

def main():
    st.title("Meta LLaMA Chatbot")
    debug_log("App started")

    # Create the chat history directory if it doesn't exist
    os.makedirs(CONFIG["chat_history_path"], exist_ok=True)

    # Initialize all session state variables at the start
    if "documents" not in st.session_state:
        st.session_state.documents = {}
    if "images" not in st.session_state:
        st.session_state.images = {}
    if "audio_transcriptions" not in st.session_state:
        st.session_state.audio_transcriptions = []
    if "chat_chain" not in st.session_state:
        st.session_state.chat_chain = None
    if "current_chat_history" not in st.session_state:
        st.session_state.current_chat_history = StreamlitChatMessageHistory(key="chat_messages")
        debug_log("Initialized new chat history")

    with st.sidebar:
        st.sidebar.title("Chat History")
        chat_sessions = ["new_session"] + [
            f.replace('.json', '') for f in os.listdir(CONFIG["chat_history_path"]) 
            if f.endswith('.json')
        ]
        debug_log(f"Available chat sessions: {chat_sessions}")

        # Handle session selection
        selected_session = st.sidebar.selectbox(
            "Select a chat session", 
            chat_sessions, 
            key="session_key",
            index=chat_sessions.index(st.session_state.get('chat_filename', 'new_session')) 
            if 'chat_filename' in st.session_state 
            else 0
        )
        debug_log(f"Selected session: {selected_session}")

        # Reset session state for new session
        if selected_session == "new_session":
            st.session_state.pop('chat_filename', None)
            st.session_state.chat_chain = None  # Reset chat chain
            st.session_state.pop('last_session', None)
            st.session_state.messages = []
            if "chat_messages" in st.session_state:
                st.session_state.chat_messages = []
            debug_log("Started a new session")

        # Initialize chat history
        chat_history = initialize_session()
        st.session_state.current_chat_history = chat_history
        st.session_state.last_session = selected_session

        # Handle file uploads
        # PDF upload
        tab_pdf, tab_images, tab_audio = st.tabs(["PDF Upload", "Image Upload", "Audio Upload"])
        with tab_pdf:
            uploaded_files = st.file_uploader("Upload PDFs", type="pdf",   accept_multiple_files=True)
            if uploaded_files:
                for i, uploaded_file in enumerate(uploaded_files):
                    unique_key = f"{uploaded_file.name}_{i}"  # Append an index to make the key unique
                    if unique_key not in st.session_state.documents:
                        with st.spinner(f"Processing PDF: {uploaded_file.name}..."):
                            vectorstore = process_pdf(uploaded_file)
                        if vectorstore is not None:
                            st.session_state.documents[unique_key] = {
                                'vectorstore': vectorstore,
                                'processed': True,
                                'filename': uploaded_file.name
                            }
                            st.success(f"PDF '{uploaded_file.name}' processed  successfully!")
                            debug_log(f"Processed PDF: {uploaded_file.name}")
                            # Reinitialize chat chain when new document is added
                            st.session_state.chat_chain = None

        # Image upload
        with tab_images:
            uploaded_images = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            if uploaded_images:
                for i, uploaded_image in enumerate(uploaded_images):
                    unique_key = f"{uploaded_image.name}_{i}"  # Append an index to make the key unique
                    if unique_key not in st.session_state.images:
                        with st.spinner(f"Processing image: {uploaded_image.name}..."):
                            try:
                                image_result = process_image(uploaded_image)
                                st.session_state.images[unique_key] = {
                                    'context': image_result,
                                    'processed': True,
                                    'filename': uploaded_image.name
                                }
                                st.success(f"Image '{uploaded_image.name}' processed successfully!")
                                debug_log(f"Processed image: {uploaded_image.name}")
                                # Reinitialize chat chain when new image is added
                                st.session_state.chat_chain = None
                            except Exception as e:
                                st.error(f"Error processing image: {str(e)}")

        # Audio upload
        with tab_audio:
            uploaded_audio = st.file_uploader("Upload Audio (MP3/WAV)", type=  ["mp3", "wav"], accept_multiple_files=False)
            if uploaded_audio:
                with st.spinner(f"Processing audio: {uploaded_audio.name}..."):
                    transcription = transcribe_audio_file(uploaded_audio)
                if transcription:
                    st.success("Audio transcription completed!")
                    st.text_area("Transcription:", transcription, height=200)
                    
                    # Ensure transcriptions are stored as strings
                    if "audio_transcriptions" not in st.session_state:
                        st.session_state.audio_transcriptions = []
                    st.session_state.audio_transcriptions.append(transcription)
                    
                    # Reinitialize chat chain when new audio is added
                    st.session_state.chat_chain = None

    # Initialize or update chat chain if needed
    if st.session_state.chat_chain is None:
        st.session_state.chat_chain = initialize_chat_chain(
            chat_history=chat_history,
            documents=st.session_state.documents,
            images=st.session_state.images,
            audio_transcripts=st.session_state.audio_transcriptions
        )
        debug_log("Chat chain initialized")

    # Display messages
    display_messages()

    # Handle user input
    prompt = st.chat_input("Type your message here")
    if prompt:
        debug_log(f"User input: {prompt}")

        st.session_state.messages.append({"type": "human", "content": prompt})
        chat_history.add_message(HumanMessage(content=prompt))

        try:
            combined_context = get_combined_context(prompt)
            response_text = st.session_state.chat_chain.invoke(combined_context)
            
            st.session_state.messages.append({"type": "assistant", "content": response_text})
            chat_history.add_message(AIMessage(content=response_text))

        except Exception as e:
            error_msg = f"Error processing message: {str(e)}"
            st.error(error_msg)
            debug_log(error_msg)

        save_chat_history(chat_history)
        debug_log("Chat history saved")

        st.rerun()

if __name__ == "__main__":
    main() 