import os
import json
import tempfile
import subprocess
import shlex
import gradio as gr
from utils.detection_utils import detect_content_from_file

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")  # default model

def generate_ollama_report(detection_result, model=OLLAMA_MODEL, timeout=20):
    """
    Calls local Ollama (ollama run <model>) with a crafted prompt.
    Returns string output or error message.
    """

    # Build prompt including the detection JSON summary
    detection_json = json.dumps(detection_result, indent=2)
    prompt = f"""You are an Ethics Advisor AI.
Given the following detection summary (JSON), produce a concise human-readable report that:
- Explains the verdict in simple terms
- Mentions confidence and suggested actions (verify source, avoid sharing, trace source, etc.)
- Provides a short recommended next step list (2-3 items)

Detection Summary:
{detection_json}

Report:"""

    try:
        # Run ollama and provide prompt via stdin.
        # Uses subprocess.run to pipe the prompt to ollama.
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout
        )
        if proc.returncode != 0:
            # Ollama returned error — include stderr for debugging
            return f"[Ollama error, rc={proc.returncode}] {proc.stderr.strip()}"
        return proc.stdout.strip()
    except FileNotFoundError:
        return "[Ollama not found] Please install Ollama and ensure `ollama` is on your PATH."
    except subprocess.TimeoutExpired:
        return "[Ollama timeout] The local model took too long to respond."
    except Exception as e:
        return f"[Ollama error] {e}"

def merge_report(detection_result, ollama_text):
    """
    Merge detection JSON into a single narrative block.
    """
    header = "[DETECTION SUMMARY]\n\n"
    det_lines = []
    # Choose a compact summary to show before the narrative
    det_type = detection_result.get("type", "unknown")
    label = detection_result.get("label", "unknown")
    confidence = detection_result.get("confidence", None)
    det_lines.append(f"Type: {det_type}")
    det_lines.append(f"Label: {label}")
    if confidence is not None:
        det_lines.append(f"Confidence: {confidence}")
    det_lines.append(f"Model: {detection_result.get('model','unknown')}")
    if detection_result.get("type") == "video":
        det_lines.append(f"Frames Sampled: {detection_result.get('frames_sampled','N/A')}")
    det_summary = "\n".join(det_lines)

    merged = f"{header}{det_summary}\n\n[AI-GENERATED REPORT]\n\n{ollama_text}"
    return merged

def analyze_file(uploaded_file, generate_report_flag=True):
    """
    Main Gradio function.
    uploaded_file is a tempfile-like object from Gradio.
    """
    if uploaded_file is None:
        return "No file uploaded."

    # Save uploaded file to temp location
    try:
        tmp_dir = tempfile.mkdtemp(prefix="df_mvp_")
        local_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(local_path, "wb") as f:
            f.write(uploaded_file.read())
    except Exception as e:
        return f"Failed to save uploaded file: {e}"

    # Run detection
    result = detect_content_from_file(local_path)

    # Compute simple risk heuristics (optional)
    if "confidence" in result and isinstance(result["confidence"], (int, float)):
        c = float(result["confidence"])
        weights = {"image": 0.5, "video": 0.6, "text": 0.3}
        w = weights.get(result.get("type"), 0.4)
        risk_score = round(c * w, 4)
        risk_level = ("Low" if risk_score < 0.25 else "Medium" if risk_score < 0.5 else "High")
        result["risk"] = {"risk_score": risk_score, "risk_level": risk_level}

    # If the user wants an Ollama report, generate it; else just return JSON pretty
    if generate_report_flag:
        ollama_text = generate_ollama_report(result)
        merged = merge_report(result, ollama_text)
        return merged
    else:
        return json.dumps(result, indent=2)

if __name__ == "__main__":
    title = "TruthShield AI: DeepFake Detection & Ethics Advisor"
    desc = (
        "Upload an image (.jpg/.png), video (.mp4/.mov/.avi), or text file (.txt). "
        "The system auto-detects the type, runs a fast DeepFake/text detector, "
        "and (optionally) generates a merged human-readable report via local Ollama (llama2)."
    )

    iface = gr.Interface(
        fn=analyze_file,
        inputs=[
            gr.File(label="Upload image / video / text file"),
            gr.Checkbox(label="Generate merged Ollama report (requires Ollama & llama2)", value=True)
        ],
        outputs=gr.Textbox(label="Merged Detection + Report"),
        title=title,
        description=desc,
        allow_flagging="never",
        analytics_enabled=False,
    )

    iface.launch()