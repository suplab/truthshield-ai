import mimetypes

def detect_media_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        if "image" in mime_type:
            return "image"
        elif "video" in mime_type:
            return "video"
        elif "text" in mime_type or "plain" in mime_type:
            return "text"
    return "unknown"