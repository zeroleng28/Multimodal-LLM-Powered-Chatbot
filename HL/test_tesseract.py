from PIL import Image
import pytesseract
import os

def test_tesseract():
    """Test Tesseract installation"""
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    print("1. Checking Tesseract path...")
    if os.path.exists(tesseract_path):
        print("✓ Tesseract executable found")
    else:
        print("✗ Tesseract executable not found")
        return False
    
    print("\n2. Setting Tesseract path...")
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    
    print("\n3. Testing Tesseract...")
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        # Create test image
        test_image = Image.new('RGB', (100, 30), color='white')
        pytesseract.image_to_string(test_image)
        print("✓ OCR test successful")
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Tesseract Installation\n" + "="*25 + "\n")
    if test_tesseract():
        print("\nTesseract is working correctly! ✅")
    else:
        print("\nTesseract installation has issues! ❌") 