from .preprocessing import preprocess
from pathlib import Path
import os
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
def image_to_text(image_bytes: bytes, model_id_override: str | None = None, temperature_override: float | None = None, config_override: str | None = None, prompt_override: str | None = None):
    #read from yaml
    config_name = config_override or "config.yaml"
    try:
        with open(Path(__file__).parent / config_name, 'r') as file:
            config = yaml.safe_load(file)
            bPreprocess = config['preprocess']
            prompt = prompt_override or config['prompt']    # override: e.g. the multipass review prompt
            model_id = model_id_override or config['model']['id']
            model_temp = config['model']['temperature'] if temperature_override is None else temperature_override
            backend = config.get('backend') or 'openai'
            if backend not in ('openai', 'gemini'):
                raise ValueError(f"backend must be 'openai' or 'gemini', got {backend!r}")
            key_env = config.get('api_key_env')
            model_url = config['model']['chat_url']
            if backend == 'gemini':
                model_url = f"{model_url.rstrip('/')}/{model_id}:generateContent"
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
    key_env = key_env or None
    api_key = os.environ.get(key_env) if key_env else None
    if key_env and not api_key:
        raise ConfigError(f"{config_name}: env var {key_env} is not set")
    if backend == "gemini" and not api_key:
        raise ConfigError(f"{config_name}: gemini backend needs api_key_env set to an env var holding the key")
    headers = {"Content-Type": "application/json"}
    if backend == "gemini":
        headers["x-goog-api-key"] = api_key
        payload = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": img}}
            ]}],
            "generationConfig": {"temperature": model_temp}
        }
    else:
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}
                    ]
                }
            ],
            "temperature": model_temp
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
        if code == 503:
            raise ServiceError(msg, "The model is busy right now. Try again in a minute, or pick a different model.")
        if code == 500:
            raise ServiceError(msg, "The model hit an internal error. Try again, or pick a different model.")
        if code == 400 and "api key not valid" in str(detail).lower():
            raise ConfigError(msg)
        if code == 400 and "modality" in str(detail).lower():    # e.g. "Image input modality is not enabled for models/..."
            raise ConfigError(msg, "This model doesn't accept images. Pick a vision-capable model, or contact the administrator.")
        if code in (401, 403, 404):
            raise ConfigError(msg)
        if code == 400:
            raise ImageError(msg, "The model rejected this image. Try a smaller or clearer photo.")
        raise ServiceError(msg)

    try:
        data = response.json()
        if backend == "gemini":
            block = (data.get("promptFeedback") or {}).get("blockReason")
            if block:
                raise ImageError(f"{model_id}: prompt blocked ({block})", "The image was blocked by Google's safety filter. Try a different image.")
            candidates = data.get("candidates") or []
            if not candidates:
                raise ServiceError(f"{model_id}: no candidates in response: {response.text[:300]}")
            cand = candidates[0]
            if cand.get("finishReason") == "SAFETY":
                raise ImageError(f"{model_id}: response blocked (finishReason SAFETY)", "The response was blocked by Google's safety filter. Try a different image.")
            parts = (cand.get("content") or {}).get("parts") or []
            gemini_text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
            data = {"choices": [{"message": {"content": gemini_text},
                                 "finish_reason": "length" if cand.get("finishReason") == "MAX_TOKENS" else "stop"}]}
        choice = data["choices"][0]
        text = choice["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as e:
        raise ServiceError(f"{model_id}: unexpected response: {response.text[:300]}") from e

    if choice.get("finish_reason") == "length":
        raise ServiceError(f"{model_id}: output truncated (finish_reason = length)", "The response was cut off. Please contact the administrator.")
    if not text.strip():
        raise EmptyResponse(f"{model_id}: empty content")

    return data