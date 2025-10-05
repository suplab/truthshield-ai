import torch
import cv2
import numpy as np
from transformers import pipeline
from PIL import Image


# ----------------------------
# Load Models
# ----------------------------

# Image DeepFake Detector (XceptionNet-style model from timm/torch hub)
image_model = torch.hub.load("huggingface/pytorch-image-models", "xception", pretrained=True)
image_model.eval()

# Text AI Detector (Zero-Shot as fallback)
text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


# ----------------------------
# Image DeepFake Detection
# ----------------------------

def preprocess_image(img):
    img = img.resize((299, 299))  # XceptionNet input size
    img = np.array(img).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Channels first
    return torch.tensor(img).unsqueeze(0)


def detect_deepfake_image(image_path):
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(img)

    with torch.no_grad():
        output = image_model(input_tensor)

    confidence = torch.softmax(output, dim=1).max().item()
    is_fake = torch.argmax(output, dim=1).item() == 1  # Assuming class 1 = Fake

    return {
        "type": "image",
        "is_fake": is_fake,
        "confidence": confidence
    }


# ----------------------------
# Video DeepFake Detection
# ----------------------------

def detect_deepfake_video(video_path, frame_limit=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    for _ in range(frame_limit):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        input_tensor = preprocess_image(img)
        frames.append(input_tensor)

    cap.release()

    if not frames:
        return {"error": "Could not read frames"}

    inputs = torch.cat(frames)
    with torch.no_grad():
        outputs = image_model(inputs)

    predictions = torch.argmax(outputs, dim=1)
    is_fake = predictions.sum().item() > (len(predictions) / 2)  # Majority vote
    confidence = torch.softmax(outputs, dim=1).mean().max().item()

    return {
        "type": "video",
        "is_fake": is_fake,
        "confidence": confidence
    }


# ----------------------------
# Text AI Detection
# ----------------------------

def detect_ai_text(text, candidate_labels=["real", "fake"]):
    result = text_classifier(text, candidate_labels)
    return {
        "type": "text",
        "is_ai_generated": result["labels"][0] == "fake",
        "confidence": result["scores"][0]
    }


# ----------------------------
# Unified Wrapper
# ----------------------------

def detect_content(input_path_or_text):
    if isinstance(input_path_or_text, str) and input_path_or_text.lower().endswith((".png", ".jpg", ".jpeg")):
        return detect_deepfake_image(input_path_or_text)

    elif isinstance(input_path_or_text, str) and input_path_or_text.lower().endswith((".mp4", ".avi", ".mov")):
        return detect_deepfake_video(input_path_or_text)

    else:
        return detect_ai_text(input_path_or_text)

