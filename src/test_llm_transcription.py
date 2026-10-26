# loops over the transcription.py's submission logic with several images
# compiles results in runs/<date_startTime>/outputs/ for each model, targets in runs/targets
# targets generated from CVL_XML_TextExtractor.py

import random
import requests
import shutil
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import yaml
from .TextExtractor import extract_text_cvl, extract_text_smhd
from .llm_transcription import image_to_text, TranscriptionError
from .alignment_test import align_score, json_align_score

image_count = int(sys.argv[1])
seed = sys.argv[2]
dataset = sys.argv[3].lower() if len(sys.argv) > 3 else "none"
runs_home = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).parent.parent / "runs"

test_models = Path(__file__).parent / "models.yaml"
with open(test_models) as f:
    file = yaml.safe_load(f)
    models = file["models"]
    context_length = file["context_length"]
    chat_url = file["chat_url"].rstrip("/")
    config_name = file["forms_config"] if dataset == "funsd" else file["transcription_config"]    # prompt/backend config passed to image_to_text
    backend = file.get("backend", "openai")
    api_key = os.environ.get(file.get("api_key_env") or "")
USE_LMSTUDIO = backend == "openai" and not api_key    # load/unload only for a local LM Studio (same gate as demo.py)

## Setup files ##
runs_path = runs_home / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
runs_path.mkdir(parents = True, exist_ok = True)
(runs_path / "images").mkdir(exist_ok = True)
(runs_path / "targets").mkdir(exist_ok = True)
(runs_path / "seed.txt").write_text(seed)

## setup data for testing ##
textTargets = True
if dataset == "cvl":
    target_path = Path(__file__).parent.parent / "data" / "cvl-database-1-1" / "testset" / "xml"
    image_path = Path(__file__).parent.parent / "data" / "cvl-database-cropped-1-1"
    target_format_files = sorted(target_path.glob("*.xml"))
    nameRemoval = "_attributes.xml" #what to remove from file name to get an id number for targets
    data_extract_text = extract_text_cvl
    image_file_type = ".tif"
    image_extension = "-cropped" + image_file_type #what to remove from image file name to get id number that matches targets
    print("Testing handwriting with CVL.")
elif dataset == "smhd":
    target_path = Path(__file__).parent.parent / "data" / "SMHD-forms" / "SMHD-forms" / "ALL" 
    image_path = Path(__file__).parent.parent / "data" / "SMHD-forms" / "SMHD-forms" / "ALL"
    target_format_files = sorted(target_path.glob("*.txt"))
    nameRemoval = ".txt"
    data_extract_text = extract_text_smhd
    image_file_type = ".jpg"
    image_extension = "" + image_file_type
    print("Testing handwriting with SMHD.")
elif dataset == "ofs":
    try:
        with open(test_models) as f:
            file = yaml.safe_load(f)
        target_path = Path(file['ofs_targets'])
        image_path = Path(file['ofs_images'])
        target_format_files = sorted(target_path.glob("*.json"))
        nameRemoval = "-p1.json"
        data_extract_text = extract_text_smhd
        image_file_type = ".png"
        image_extension = "-p1" + image_file_type
        print("!!! WARNING: Ensure you are testing with self hosted models so the OFS data is not used for AI training. 10 second countdown. Current config: " + str(test_models) )
        time.sleep(10)
        print("Testing handwriting with OFS.")
    except exception as e:
        print("ofs_target and ofs_images must be set in the models.yaml file to test on secure OFS data.")
        sys.exit(1)
elif dataset == "funsd":
    textTargets = False
    target_path = Path(__file__).parent.parent / "data" / "funsd" / "dataset" / "testing_data" / "annotations"
    image_path = Path(__file__).parent.parent / "data" / "funsd" / "dataset" / "testing_data" / "images"
    target_format_files = sorted(target_path.glob("*.json"))
    nameRemoval = ".json"
    data_extract_text = None
    image_file_type = ".png"
    image_extension = "" + image_file_type
    print("Testing Forms with FUNSD")
else:
    print("ERROR: Missing data string as arguement 3. Possible options: cvl, smhd, funsd")
    sys.exit(0)

#get random images, store in images and targets
targets = [] #list of strings
image_paths = [] #list of image paths

selected = random.Random(int(seed)).sample(target_format_files, image_count)
for i, file in enumerate(selected, start=1):
    # extract text from xml, save to runs/targets
    page_id = file.name.replace(nameRemoval, "")
    if textTargets:
        target = data_extract_text(file)
        targets.append(target)
        (runs_path / "targets" / f"trgt{i:03d}_{page_id}.txt").write_text(target, encoding="utf-8")
    # get associated source image, copy to runs/images
    src_img = image_path / f"{page_id}{image_extension}"
    if not src_img.exists():
        raise FileNotFoundError(f"No cropped image for {page_id}: {src_img}")
    else:
        dst_img = runs_path / "images" / f"img{i:03d}_{page_id}{image_file_type}"
        image_paths.append(dst_img)
        shutil.copy(src_img, dst_img)


## model loading and unloading ##
def load_model(model_id: str, context_length: int = 16384) -> None:
    """Unload any existing copy, then load fresh with the context length we want."""
    info = requests.get(f"{chat_url}/api/v1/models", timeout=30).json()["models"]
    for m in info:
        if m["key"] == model_id:
            for inst in m["loaded_instances"]:
                requests.post(f"{chat_url}/api/v1/models/unload",
                              json={"instance_id": inst["id"]}, timeout=120)

    r = requests.post(f"{chat_url}/api/v1/models/load",
                      json={"model": model_id, "context_length": context_length,
                            "echo_load_config": True},
                      timeout=600)
    r.raise_for_status()
    print(f"Loaded {model_id}: {r.json()['load_config']}")


def unload_model(model_id: str) -> None:
    requests.post(f"{chat_url}/api/v1/models/unload", json={"instance_id": model_id}, timeout=120)

def unload_all() -> None:
    for m in requests.get(f"{chat_url}/api/v1/models", timeout=30).json()["models"]:
        for inst in m["loaded_instances"]:
            requests.post(f"{chat_url}/api/v1/models/unload", json={"instance_id": inst["id"]}, timeout=120)

## retry on rate limits / temporary server errors so the test finishes with every image ##
RETRY_CODES = ("HTTP 429", "HTTP 500", "HTTP 503")    # rate limited, internal error, high demand
RETRY_WAITS = [15, 30, 60, 120, 240]                  # seconds before each retry (~8 min total), then give up

def transcribe_with_retry(image_bytes: bytes, model_id: str, temperature: float) -> dict:
    for wait in RETRY_WAITS + [None]:
        try:
            return image_to_text(image_bytes, model_id, temperature, config_name)
        except TranscriptionError as e:
            if wait is None or not any(code in str(e) for code in RETRY_CODES):
                raise
            print(f"  {str(e)[:120]} -- retrying in {wait}s")
            time.sleep(wait)

## main logic ##
print(f"Models: {test_models.name}  config: {config_name}  backend: {backend}  LM Studio load/unload: {USE_LMSTUDIO}")
if USE_LMSTUDIO:
    unload_all()
transcriptions = []
scores = {}
for m in models:
    model_dir = runs_path / "outputs" / m["id"]
    model_dir.mkdir(exist_ok = True, parents = True)
    if USE_LMSTUDIO:
        try:
            load_model(m["id"], context_length = context_length)
        except requests.RequestException as e:
            print(f"Could not load {m['id']}, skipping: {e}")
            (model_dir / "error.txt").write_text(f"load failed: {e}\n", encoding="utf-8")
            continue
    current_scores = []
    for i, img in enumerate(image_paths, start = 1):
        with open(img, "rb") as image_file:
            image_bytes = image_file.read()
        try:
            transcript = transcribe_with_retry(image_bytes, m["id"], m["temperature"])["choices"][0]["message"]["content"]
            transcriptions.append(transcript)
            page_id = img.stem.split("_", 1)[1]
            (model_dir / f"out{i:03d}_{page_id}.txt").write_text(transcript, encoding="utf-8")
            if textTargets:
                score = align_score(targets[i - 1], transcript, verbose = False)
            else:
                score = json_align_score(transcript)
            current_scores.append((img.stem, score))
            if textTargets:
                if score["cer"] is None:
                    print(f"Completed image {i}  (blank target, not scored)")
                else:
                    print(f"Completed image {i}  CER {score['cer']:.1%}  WER {score['wer']:.1%}")
        except TranscriptionError as e:
            print(f"Error on model {m['id']}, image {i}: {e}")
            with open(model_dir / "error.txt", "a", encoding="utf-8") as f:
                f.write(f"image {i} ({img.name}): {e}\n")
    if USE_LMSTUDIO:
        unload_model(m["id"])
    scores[m["id"]] = current_scores

scores_file = runs_path / "scores.csv"
with open(scores_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if textTargets:
        writer.writerow(["model", "image", "cer", "wer"])
        for model_id, results in scores.items():
            for image_name, s in results:
                writer.writerow([model_id, image_name, s["cer"], s["wer"]])
    else:
        writer.writerow(["model", "image", "json_format_error"])
        for model_id, results in scores.items():
            for image_name, error_code in results:
                writer.writerow([model_id, image_name, error_code])