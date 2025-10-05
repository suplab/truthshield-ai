
def risk_scoring(media_type, detection_score):
    weights = {"image": 0.4, "video": 0.5, "text": 0.3}
    risk = detection_score * weights.get(media_type, 0.3)
    risk_level = "Low" if risk < 0.3 else "Medium" if risk < 0.6 else "High"
    return {"risk_score": round(risk, 2), "risk_level": risk_level}
