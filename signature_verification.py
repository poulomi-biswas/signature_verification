import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import hashlib
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

# Step 1: Preprocess the image (thresholding and noise removal)
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return clean

# Step 2: Find the largest contour (signature area)
def find_signature_contours(thresh_image):
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    if signature_contours:
        return max(signature_contours, key=cv2.contourArea)
    else:
        return None

# Step 3: Draw bounding box around the signature
def draw_bounding_box(image, contour):
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image, (x, y, w, h)
    else:
        return image, None

# Step 4: Crop the signature region from the image
def crop_signature(image, bounding_box):
    if bounding_box is not None:
        x, y, w, h = bounding_box
        cropped_image = image[y:y+h, x:x+w]
        return cropped_image
    return None

# Step 5: Remove extra white spaces using Morphological Transformations
def remove_white_space(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (25, 25), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)
    
    coords = cv2.findNonZero(close)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y+h, x:x+w]

# Step 6: Compare signatures using SSIM
def compare_signatures(reference_signature, test_signature):
    gray_reference = cv2.cvtColor(reference_signature, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_signature, cv2.COLOR_BGR2GRAY)
    
    resized_reference = cv2.resize(gray_reference, (100, 100))
    resized_test = cv2.resize(gray_test, (100, 100))
    
    score, _ = ssim(resized_reference, resized_test, full=True)
    return score

# Step 7: Compute a hash for both reference and test signatures
def compute_hash(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (100, 100))
    img_bytes = resized_image.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()

# Step 8: Digital Signature Generation and Verification
def generate_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = private_key.public_key()
    return private_key, public_key

def sign_hash(private_key, message_hash):
    signature = private_key.sign(
        message_hash.encode(),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return signature

def verify_signature(public_key, signature, message_hash):
    try:
        public_key.verify(
            signature,
            message_hash.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True
    except:
        return False

# Utility: Show images using matplotlib
def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Main Execution
image = cv2.imread('C:/Users/PRADIP/Downloads/doc1.png')
reference_signature = cv2.imread('C:/Users/PRADIP/Downloads/doc1f.png')

# Preprocess and clean up both original and reference images
preprocessed_image = preprocess_image(image)
largest_contour = find_signature_contours(preprocessed_image)
_, bounding_box = draw_bounding_box(image.copy(), largest_contour)
cropped_signature = crop_signature(image, bounding_box)
cleaned_signature = remove_white_space(cropped_signature)

preprocessed_reference = preprocess_image(reference_signature)
reference_contour = find_signature_contours(preprocessed_reference)
_, reference_bounding_box = draw_bounding_box(reference_signature.copy(), reference_contour)
cropped_reference_signature = crop_signature(reference_signature, reference_bounding_box)
cleaned_reference_signature = remove_white_space(cropped_reference_signature)

# Display images
show_image(image, "Original Image")
show_image(preprocessed_image, "Preprocessed Image")
show_image(cropped_signature, "Detected and Cropped Signature")
show_image(cleaned_signature, "Cleaned Signature")

# SSIM Score
ssim_score = compare_signatures(cleaned_reference_signature, cleaned_signature)
print(f"SSIM Score: {ssim_score}")

# Compute Hashes
reference_hash = compute_hash(cleaned_reference_signature)
test_hash = compute_hash(cleaned_signature)
print(f"Reference Hash: {reference_hash}")
print(f"Test Hash: {test_hash}")

# Digital Signature Generation and Verification
private_key, public_key = generate_keys()
signature = sign_hash(private_key, reference_hash)
print(f"Digital Signature for Reference Hash: {signature}")

# Verify the signature
is_verified = verify_signature(public_key, signature, test_hash)
print(f"Signature Verified: {is_verified}")

# Final Verification
if ssim_score > 0.8 and is_verified:
    print("Signatures match with high confidence.")
else:
    print("Signatures do not match.")
