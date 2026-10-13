from .preprocessing import preprocess
from pathlib import Path
import yaml
import os
import requests
import json
import base64
from PIL import Image
import io

class TranscriptionError(Exception):
    """Raised when the model request fails."""

def image_to_text(image_bytes: bytes, model_id_override: str | None = None, temperature_override: float | None = None):
    #read from yaml
    with open(Path(__file__).parent / "llm_transcription_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    prompt = config["prompt"]
    bPreprocess = config["preprocess"]
    if model_id_override is None:
        model_id = config["model"]["id"]
    else:
        model_id = model_id_override
    if temperature_override is None:
        model_temp = config["model"]["temperature"]
    else:
        model_temp = temperature_override
    model_url = config["model"]["chat_url"]
    model_timeout = config["model"]["timeout"]
    
    #preprocess image
    #convert to png
    img = Image.open(io.BytesIO(image_bytes))
    #additional preprocessing - removing lines, etc - as mentioned on PPT slide
    print("Preprocessing enabled:", bPreprocess)
    if bPreprocess:
        img = preprocess(img)
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    img = buf.getvalue()
    #all require base64 encoding
    img = base64.b64encode(img).decode("utf-8")

    #api request to some openAI endpoint
    json = {
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
        response = requests.post(model_url, json = json, headers = headers, timeout = model_timeout)
        response.raise_for_status()
        print("Transcription completed.")
    except requests.HTTPError:
        try:
            detail = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            detail = response.text[:500]
        raise TranscriptionError(f"{model_id}: HTTP {response.status_code}: {detail}") from None
    except requests.RequestException as e:
        raise TranscriptionError(f"{model_id}: request failed: {e}") from None
        
    return response.json()