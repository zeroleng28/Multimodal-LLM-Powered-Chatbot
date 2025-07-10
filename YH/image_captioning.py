
import streamlit as st
from PIL import Image
import requests
import io



class ImageToTextFallback:
    def __init__(self, api_token, model_url="https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"):
        """
        Initialize image-to-text fallback with Hugging Face API
        
        Args:
            api_token (str): Hugging Face API token
            model_url (str, optional): URL of the image captioning model
        """
        self.API_URL = model_url
        self.headers = {"Authorization": f"Bearer {api_token}"}
    
    def query(self, filename):
        """
        Direct query method for image-to-text conversion
        
        Args:
            filename (str or file-like object): Path to image or file object
        
        Returns:
            dict: API response
        """
        try:
            # Handle different input types
            if isinstance(filename, str):
                # If it's a file path
                with open(filename, "rb") as f:
                    data = f.read()
            elif hasattr(filename, 'getvalue'):
                # If it's a Streamlit uploaded file
                data = filename.getvalue()
            elif isinstance(filename, Image.Image):
                # If it's a PIL Image
                img_byte_arr = io.BytesIO()
                filename.save(img_byte_arr, format='PNG')
                data = img_byte_arr.getvalue()
            else:
                raise ValueError("Unsupported input type")
        
            # Make API request
            response = requests.post(
                self.API_URL, 
                headers=self.headers, 
                data=data
            )
            
            # Return JSON response
            return response.json()
        
        except Exception as e:
            st.error(f"Error in image-to-text query: {e}")
            return None

def image_to_text_fallback(images, api_token):
    """
    Fallback method to generate captions for images using Hugging Face API
    
    Args:
        images (list): List of image files
        api_token (str): Hugging Face API token
    
    Returns:
        str: Concatenated image captions
    """
    # Initialize fallback
    fallback = ImageToTextFallback(api_token)
    
    captions = []
    for img in images:
        try:
            # Perform direct query
            output = fallback.query(img)
            
            # Extract caption from response
            if output and isinstance(output, list) and len(output) > 0:
                caption = output[0].get('generated_text', 'No caption generated')
                captions.append(f"Image Caption: {caption}")
            else:
                captions.append(f"No caption generated for {img.name}")
        
        except Exception as img_error:
            st.warning(f"Could not process image {img.name}: {img_error}")
    
    # Return concatenated captions
    return "\n\n".join(captions) if captions else ""


def preprocess_for_image_to_text(image):
    """
    Preprocess image to improve image-to-text model performance.
    
    Args:
        image (PIL.Image): Input image
    
    Returns:
        PIL.Image: Preprocessed image
    """
    # Resize image if too large
    max_size = (800, 800)  # Adjust as needed
    image.thumbnail(max_size, Image.LANCZOS)
    
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image