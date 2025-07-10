# audio_transcription.py
import os
import streamlit as st
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration


os.environ['PATH'] += os.pathsep + r'C:\ffmpeg' 

@st.cache_resource
def load_asr_pipeline(model_name="openai/whisper-base"):
    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.config.forced_decoder_ids = None

    # Create the pipeline from the model and processor components
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True
    )
    return asr

def transcribe_audio_files(audio_files, model_name="openai/whisper-base"):
    asr = load_asr_pipeline(model_name=model_name)
    all_transcriptions = []
    os.makedirs("temp", exist_ok=True)

    for idx, audio_file in enumerate(audio_files,start=1):
        # Save the file locally
        audio_bytes = audio_file.read()
        temp_audio_path = os.path.join("temp", audio_file.name)
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)

        # Transcribe using the pipeline
        result = asr(temp_audio_path)
        transcription = result["text"].strip()
        if transcription:
            formatted_transcription = f"audiofile {idx}:{audio_file.name}, Transcription:{transcription}"
            all_transcriptions.append(formatted_transcription)
        else:
            st.warning(f"No transcription result for {audio_file.name}")

    return "\n".join(all_transcriptions)
