import gradio as gr
import os
from dotenv import load_dotenv
from utils.media_utils import detect_media_type
from utils.detection_utils import detect_deepfake, detect_ai_text
from utils.risk_utils import risk_scoring
import openai

# Load OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_report(media_type, detection_result, risk_result, content_summary=""):
    prompt = f"""
    You are an Ethics Advisor AI.
    Analyze the following {media_type} content:

    Detection Result: {detection_result}
    Risk Assessment: {risk_result}
    Content Summary: {content_summary}

    Generate a structured human-readable report explaining:
    - Whether the content is AI-generated
    - Potential social, political, financial impact
    - Recommendations for action
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3
    )
    return response['choices'][0]['message']['content']

def analyze_content(file, text_input):
    if file is not None:
        media_type = detect_media_type(file.name)
        detection_result = detect_deepfake(file.name, media_type)
        risk_result = risk_scoring(media_type, detection_result['score'])
        report = generate_report(media_type, detection_result, risk_result)
    elif text_input:
        media_type = "text"
        detection_result = detect_ai_text(text_input)
        risk_result = risk_scoring(media_type, detection_result['score'])
        report = generate_report(media_type, detection_result, risk_result, text_input)
    else:
        return "No input provided."
    return report

ui = gr.Interface(
    fn=analyze_content,
    inputs=[gr.File(type="file"), gr.Textbox(placeholder="Paste text here")],
    outputs=gr.Textbox(label="Ethics & Risk Report"),
    title="DeepFake Detection & Ethics Advisor",
    description="Upload an image/video or enter text to detect AI-generated content and assess risks."
)

if __name__ == "__main__":
    ui.launch()
