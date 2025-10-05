import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline

# ----------------------------
# Lightweight MesoNet (MesoInception4 variant)
# ----------------------------
class MesoBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class MesoInception4(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            MesoBlock(3, 8, k=3),
            MesoBlock(8, 8, k=3),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            MesoBlock(8, 16, k=3),
            MesoBlock(16, 16, k=3),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            MesoBlock(16, 32, k=3),
            MesoBlock(32, 32, k=3),
            nn.MaxPool2d(2)
        )
        self.block4 = nn.Sequential(
            MesoBlock(32, 64, k=3),
            MesoBlock(64, 64, k=3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, 2)  # 0=Real, 1=Fake

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------------------
# Model loading / device
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MESO_WEIGHTS_PATH = os.path.join("models", "meso_inception4.pth")

meso_model = MesoInception4().to(DEVICE)
meso_model.eval()

if os.path.exists(MESO_WEIGHTS_PATH):
    try:
        state = torch.load(MESO_WEIGHTS_PATH, map_location=DEVICE)
        meso_model.load_state_dict(state)
        print("Loaded MesoNet weights from", MESO_WEIGHTS_PATH)
    except Exception as e:
        print("Could not load MesoNet weights:", e)
        # fallback to randomly init (still works as pipeline but not accurate)

# ----------------------------
# Transforms
# ----------------------------
transform = T.Compose([
    T.Resize((256, 256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ----------------------------
# Text detection pipeline (Hugging Face)
# ----------------------------
text_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ----------------------------
# Helpers
# ----------------------------
def score_to_label(score, threshold=0.5):
    if score >= threshold:
        return "Likely AI-Generated Face"
    else:
        return "Real Human"

# ----------------------------
# Image detection
# ----------------------------
def detect_deepfake_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Could not open image: {e}"}

    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = meso_model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        prob_fake = float(probs[1])
        label = score_to_label(prob_fake, threshold=0.5)
        return {
            "type": "image",
            "label": label,
            "confidence": round(prob_fake, 4),
            "model": "MesoInception4"
        }

# ----------------------------
# Video detection
# ----------------------------
def detect_deepfake_video(video_path, frame_limit=30, sample_rate=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Could not open video file"}

    frames_probs = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = 0
    sampled = 0
    while sampled < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            x = transform(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = meso_model(x)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                frames_probs.append(float(probs[1]))
            sampled += 1
        idx += 1

    cap.release()

    if len(frames_probs) == 0:
        return {"error": "No frames sampled from video"}

    avg_prob_fake = float(np.mean(frames_probs))
    label = score_to_label(avg_prob_fake, threshold=0.5)
    return {
        "type": "video",
        "label": label,
        "confidence": round(avg_prob_fake, 4),
        "frames_sampled": len(frames_probs),
        "total_frames": total_frames,
        "model": "MesoInception4"
    }

# ----------------------------
# Text detection
# ----------------------------
def detect_ai_text_file(text_content):
    candidate_labels = ["AI-generated", "Human-written"]
    try:
        result = text_classifier(text_content, candidate_labels)
        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        is_ai = top_label == "AI-generated"
        label = "Likely AI-Generated Text" if is_ai else "Human-Written Text"
        return {
            "type": "text",
            "label": label,
            "confidence": round(top_score, 4),
            "model": "zero-shot-bart-large-mnli"
        }
    except Exception as e:
        return {"error": f"text classifier error: {e}"}

# ----------------------------
# Unified wrapper
# ----------------------------
def detect_content_from_file(file_path):
    if not file_path or not isinstance(file_path, str):
        return {"error": "Invalid file path"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        return detect_deepfake_image(file_path)
    if ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
        return detect_deepfake_video(file_path)
    if ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                txt = f.read()
            return detect_ai_text_file(txt)
        except Exception as e:
            return {"error": f"Could not read text file: {e}"}

    try:
        return detect_deepfake_image(file_path)
    except Exception:
        return {"error": f"Unsupported file extension: {ext}"}