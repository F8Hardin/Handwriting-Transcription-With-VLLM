from .preprocessing import preprocess
from pathlib import Path
import yaml
import requests
import base64
from PIL import Image
import io

## Error handling ##
class TranscriptionError(Exception):
    user_msg = "Something went wrong. Please contact the administrator."
    def __init__(self, detail: str, user_msg: str | None = None):
        super().__init__(detail)
        if user_msg:
            self.user_msg = user_msg

class ConfigError(TranscriptionError):
    user_msg = "The app is misconfigured. Please contact the administrator."

class ImageError(TranscriptionError):
    user_msg = "Couldn't read that image. Please retake the photo or upload a different file."

class ConnectionFailure(TranscriptionError):
    user_msg = "Can't reach the transcription service. Check your connection and try again."

class ServiceError(TranscriptionError):
    user_msg = "The transcription service had a problem. Try again in a minute; if it keeps happening, contact the administrator."

class EmptyResponse(TranscriptionError):
    user_msg = "No text was found. Try better lighting, move closer, and keep the page flat and fully in frame."

## image to text ##
def image_to_text(image_bytes: bytes, model_id_override: str | None = None, temperature_override: float | None = None, config_override: str | None = None):
    #read from yaml
    config_name = config_override or "config.yaml"
    try:
        with open(Path(__file__).parent / config_name, 'r') as file:
            config = yaml.safe_load(file)
            bPreprocess = config['preprocess']
            prompt = config['prompt']
            model_id = model_id_override or config['model']['id']
            model_temp = config['model']['temperature'] if temperature_override is None else temperature_override
            model_url = config['model']['chat_url']
            model_timeout = config['model']['timeout']
    except Exception as e:
        raise ConfigError(f"{config_name}: {e!r}") from e
    
    #preprocess image
    try:
        #convert to png
        img = Image.open(io.BytesIO(image_bytes))
        #additional preprocessing - removing lines, etc - as mentioned on PPT slide
        if bPreprocess:
            img = preprocess(img)
        buf = io.BytesIO()
        img.save(buf, format = "PNG")
        img = buf.getvalue()
        #all require base64 encoding
        img = base64.b64encode(img).decode("utf-8")
    except Exception as e:
      raise ImageError(f"image decode/preprocess failed: {e!r}") from e

    #api request to some openAI endpoint
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img}"
                        }
                    }
                ]
            }
        ],
        "temperature": model_temp
    }

    headers = {
        "Content-Type": "application/json"
    }

    print("Sending image for transcription...")
    try:
        response = requests.post(model_url, json = payload, headers = headers, timeout = model_timeout)
        print("Transcription completed.")
    except requests.Timeout as e:
        raise ConnectionFailure(f"{model_id}: timed out after {model_timeout}s", "The model took too long to respond. Try again or pick a smaller model.") from e
    except requests.RequestException as e:
        raise ConnectionFailure(f"{model_id}: {e!r}") from e

    if not response.ok:
        try:
            detail = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = response.text[:500]
        code = response.status_code
        msg = f"{model_id}: HTTP {code}: {detail}"
        if "context size" in str(detail).lower():    # LM Studio reports this as HTTP 400
            raise ServiceError(msg, "The model's context size was exceeded. Please contact the administrator.")
        if code == 429:
            raise ServiceError(msg, "The service is busy. Wait a moment and try again.")
        if code in (401, 403, 404):
            raise ConfigError(msg)
        if code == 400:
            raise ImageError(msg, "The model rejected this image. Try a smaller or clearer photo.")
        raise ServiceError(msg)

    try:
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ServiceError(f"{model_id}: unexpected response: {response.text[:300]}") from e

    if choice.get("finish_reason") == "length":
        raise ServiceError(f"{model_id}: output truncated (finish_reason = length)", "The response was cut off. Please contact the administrator.")
    if not text.strip():
        raise EmptyResponse(f"{model_id}: empty content")

    return data