# ✅ Deep Fake Detection & Ethics Advisor

**Objective:** Autonomous agent that detects AI-generated content, flags potential misinformation, and produces explainable reports.

## MVP Features:

- Upload image, video, or text content.
- Detect if content is AI-generated.
- Assess potential impact (e.g., political, social, financial).
- Generate a human-readable report with risk scoring.

## Agentic Workflow:

1. User uploads media → Agent extracts features (frames, metadata, text).
2. Agent runs AI detection models (image/video: DeepFake detection; text: LLM content classification).
3.Agent cross-references content context (social media trends, news events) for risk scoring.
4. Agent generates a report with explanations and recommendations.
5. Optional: Agent flags suspicious content on social media using APIs.

## Tech Stack:

- Models: DeepFaceLab / FaceForensics++, HuggingFace transformers for text detection.
- LLM: GPT-4/GPT-4o-mini for report generation.
- Database: vector DB for storing content fingerprints.
- Frontend: Gradio for rapid UI.
- Extras: Social media APIs for context enrichment.

## Hackathon Roadmap:

- Build upload interface → detect media type.
- Integrate pre-trained DeepFake detection model + text classifier.
- Generate a structured report using GPT API.
- Deploy on Streamlit/Gradio.
- Bonus: Connect to a social media feed for live scanning.

## Architecture Overview

```
User Upload (Image / Video / Text)
            │
            ▼
   Media Type Detection
            │
  ┌─────────┴─────────┐
  │                   │
Image/Video          Text
Detection             Detection
(DeepFake Model)     (LLM/Text Classifier)
  │                   │
  └─────────┬─────────┘
            ▼
      Context Analysis
  (Trends, News, Social Impact)
            │
            ▼
   Risk Scoring & Explanation
            │
            ▼
     Generate Report (GPT-4)
            │
            ▼
        Gradio UI
```

## Project Structure

```
truthshield-ai/
├── README.md
├── requirements.txt
├── app.py
├── models/
│   └── placeholder_models.py
├── utils/
│   ├── media_utils.py
│   ├── detection_utils.py
│   └── risk_utils.py
├── examples/
│   ├── sample_image.jpg
│   ├── sample_video.mp4
│   └── sample_text.txt
└── config.env
```

## Setup

```bash
git clone <repo_url>
cd truthshield_ai

python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

python app.py
```

Open the Gradio link in your browser and test with example media in `examples/`.