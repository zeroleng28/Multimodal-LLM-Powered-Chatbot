import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
import io
import tempfile
import os
import time

# Configure FFmpeg path
os.environ['PATH'] += os.pathsep + r'C:\ffmpeg'

def setup_ffmpeg():
    """Setup FFmpeg path and verify installation"""
    try:
        # Test FFmpeg by creating a silent audio segment
        AudioSegment.silent(duration=1000)
        print("FFmpeg is configured correctly")
        return True
    except Exception as e:
        print(f"FFmpeg error: {str(e)}")
        st.error("FFmpeg not found! Please ensure FFmpeg is properly installed.")
        return False

def convert_mp3_to_wav(audio_file):
    """Convert MP3 file to WAV format"""
    try:
        print(f"Starting conversion of {audio_file.name}...")
        
        # Create a temporary file for the MP3
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_mp3:
            # Reset file pointer and write MP3 data
            audio_file.seek(0)
            temp_mp3.write(audio_file.read())
            temp_mp3.flush()
            
            print(f"Temporary MP3 file created: {temp_mp3.name}")
            
            try:
                # Load the MP3 file
                audio = AudioSegment.from_mp3(temp_mp3.name)
                print("MP3 file loaded successfully")
                
                # Create WAV bytes
                wav_bytes = io.BytesIO()
                audio.export(wav_bytes, format="wav", parameters=["-ac", "1", "-ar", "16000"])
                wav_bytes.seek(0)
                
                print("WAV conversion successful")
                return wav_bytes.read()
                
            finally:
                # Clean up temporary MP3 file
                try:
                    os.unlink(temp_mp3.name)
                    print("Cleaned up temporary MP3 file")
                except Exception as e:
                    print(f"Error cleaning up MP3 file: {str(e)}")
        
    except Exception as e:
        print(f"Error converting MP3 to WAV: {str(e)}")
        print(f"File name: {audio_file.name}")
        print(f"File size: {audio_file.tell()} bytes")
        
        # Try alternative conversion method
        try:
            print("Attempting alternative conversion method...")
            audio_file.seek(0)
            audio = AudioSegment.from_file(audio_file, format="mp3")
            
            wav_bytes = io.BytesIO()
            audio.export(wav_bytes, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            wav_bytes.seek(0)
            
            print("Alternative conversion successful")
            return wav_bytes.read()
            
        except Exception as alt_e:
            print(f"Alternative conversion failed: {str(alt_e)}")
            return None

def verify_audio_file(audio_file):
    """Verify that the audio file is valid"""
    try:
        # Check file size
        audio_file.seek(0, 2)  # Seek to end
        file_size = audio_file.tell()
        audio_file.seek(0)  # Reset to start
        
        if file_size == 0:
            print("Error: File is empty")
            return False
            
        # Check file extension
        if not audio_file.name.lower().endswith(('.mp3', '.wav')):
            print(f"Error: Unsupported file format: {audio_file.name}")
            return False
            
        # Read first few bytes to check if file is corrupted
        header = audio_file.read(256)
        audio_file.seek(0)
        
        if len(header) == 0:
            print("Error: Cannot read file content")
            return False
            
        print(f"File verification successful: {audio_file.name} ({file_size} bytes)")
        return True
        
    except Exception as e:
        print(f"Error verifying file: {str(e)}")
        return False

def convert_audio_to_text(audio_files):
    """Convert audio files to text using Speech Recognition"""
    # Verify FFmpeg installation
    if not setup_ffmpeg():
        return ""
    
    texts = []
    recognizer = sr.Recognizer()
    
    for audio_file in audio_files:
        temp_path = None
        try:
            # Show processing status
            st.info(f"Processing {audio_file.name}...")
            print(f"Processing audio file: {audio_file.name}")
            
            # Verify audio file first
            if not verify_audio_file(audio_file):
                raise Exception(f"Invalid audio file: {audio_file.name}")
            
            # Create temporary file with unique name
            temp_path = tempfile.mktemp(suffix='.wav')
            
            # Convert to WAV if needed
            if audio_file.name.lower().endswith('.mp3'):
                wav_bytes = convert_mp3_to_wav(audio_file)
                if wav_bytes is None:
                    raise Exception(f"Failed to convert {audio_file.name} to WAV")
                
                with open(temp_path, 'wb') as f:
                    f.write(wav_bytes)
            else:
                # For WAV files, copy directly
                audio_file.seek(0)
                with open(temp_path, 'wb') as f:
                    f.write(audio_file.read())
            
            print(f"Temporary WAV file created: {temp_path}")
            
            # Process the audio file
            with sr.AudioFile(temp_path) as source:
                print("Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source)
                
                print("Recording audio...")
                audio = recognizer.record(source)
                
                print("Converting speech to text...")
                text = recognizer.recognize_google(audio)
                
                if text.strip():
                    texts.append(text)
                    st.success(f"Successfully transcribed {audio_file.name}")
                    print(f"Transcription: {text[:100]}...")
                else:
                    st.warning(f"No speech detected in {audio_file.name}")
                    print("No speech detected")
            
        except sr.UnknownValueError:
            st.warning(f"Could not understand audio in {audio_file.name}")
            print("Speech recognition could not understand the audio")
        except sr.RequestError as e:
            st.error(f"Error with speech recognition service: {str(e)}")
            print(f"Speech recognition service error: {str(e)}")
        except Exception as e:
            st.error(f"Error processing {audio_file.name}: {str(e)}")
            print(f"Error details: {type(e).__name__} - {str(e)}")
        
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    # Wait a moment to ensure file is not in use
                    time.sleep(0.1)
                    os.remove(temp_path)
                    print(f"Cleaned up temporary file: {temp_path}")
                except Exception as e:
                    print(f"Error cleaning up temporary file: {str(e)}")
            
            # Reset file pointer
            audio_file.seek(0)
    
    return "\n".join(texts)

def process_audio_files(audio_files):
    """Process audio files and return transcribed text"""
    if not audio_files:
        st.warning("Please upload at least one audio file")
        return ""
    
    print(f"Processing {len(audio_files)} audio files...")
    with st.spinner("Processing Audio Files..."):
        audio_text = convert_audio_to_text(audio_files)
        if audio_text:
            # Show transcribed text
            with st.expander("View Transcribed Text"):
                st.write(audio_text)
            st.success("Audio processed successfully!")
            print("Audio processing completed successfully")
            return audio_text
        else:
            st.warning("No speech could be transcribed from the audio files")
            print("No text was transcribed from the audio files")
            return ""

# Test FFmpeg configuration if running directly
if __name__ == "__main__":
    if setup_ffmpeg():
        print("FFmpeg is properly configured!")
    else:
        print("FFmpeg configuration failed!")