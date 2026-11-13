import io
import json
import re
from pathlib import Path
import yaml
import jiwer
from PIL import Image
from .llm_transcription import image_to_text

REVIEW_PROMPT = """{task}

A previous pass already processed this image. Its output is below. Check it against the image line by line, fix any errors \
(misread words, missing or extra text, wrong fields), and return the complete corrected output in exactly the same format. \
Return only the output, with no explanations.

Previous output:
{previous}"""

def format_output(text: str, as_json: bool) -> tuple[str, bool]:
    #returns (text to show, ok): forms strip code fences and pretty-print JSON, ok = valid JSON
    if not as_json:
        return text, True
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.dumps(json.loads(cleaned), indent=2, ensure_ascii=False), True
    except json.JSONDecodeError:
        return "Not valid JSON:\n\n" + text, False

def config_prompt(config: str) -> str:
    #returns the task prompt from the yaml config file next to this module
    with open(Path(__file__).parent / config, encoding="utf-8") as f:
        return yaml.safe_load(f)["prompt"]

def single_send(img: Image.Image, model_id, temperature, config: str, prompt: str | None = None) -> dict:
    #returns json with fields: final_response (raw model text from one image_to_text call)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"final_response": image_to_text(buf.getvalue(), model_id, temperature, config, prompt)["choices"][0]["message"]["content"]}

def multipass(img: Image.Image, model_id, temperature, config: str, as_json: bool) -> dict:
    #returns json with fields: final_response, original_response, percent_changed, similarity (0-100, pass 1 vs pass 2), ok
    first = single_send(img, model_id, temperature, config)["final_response"]
    second = single_send(img, model_id, temperature, config, REVIEW_PROMPT.format(task=config_prompt(config), previous=first))["final_response"]
    (first_shown, _), (second_shown, ok) = format_output(first, as_json), format_output(second, as_json)
    word_length_1 = len(first_shown.split())
    max_word_len = max(word_length_1, len(second_shown.split()))
    if max_word_len > 0:
        changed = min(1.0, jiwer.wer(first_shown, second_shown) * word_length_1 / max_word_len)
    else:
        changed = 0.0
    return {"final_response": second_shown, "original_response": first_shown, "percent_changed": changed, "similarity": (1 - changed) * 100, "ok": ok}

def compare_responses(texts: list[str]) -> dict:
    #returns json with fields: similarities (% per text vs the rest), distances (total word edits per text), best_index (lowest total distance, first wins ties), similarity (mean of similarities)
    comparisons_count = len(texts) - 1 if len(texts) > 1 else 1
    distances = []
    similarities = []
    for i in range(len(texts)):
        cumulative_dist = 0
        total_similarity = 0
        for j in range(len(texts)):
            if i != j:
                word_length_i = len(texts[i].split())
                curr_dist = jiwer.wer(texts[i], texts[j]) * word_length_i
                cumulative_dist += curr_dist
                max_word_len = max(word_length_i, len(texts[j].split()))
                if max_word_len > 0:
                    total_similarity += max(0.0, 1.0 - (curr_dist / max_word_len))
                else:
                    total_similarity += 1.0
        similarities.append((total_similarity / comparisons_count) * 100)
        distances.append(cumulative_dist)
    return {"similarities": similarities, "distances": distances, "best_index": distances.index(min(distances)), "similarity": sum(similarities) / len(similarities)}

def multisend(img: Image.Image, model_id, temperature, config: str, as_json: bool, num_send: int) -> dict:
    #returns json with fields: final_response, responses ([text, similarity %] per run), best_index, distances, similarity (0-100, mean of the per-run similarities), ok
    formatted = [format_output(single_send(img, model_id, temperature, config)["final_response"], as_json) for _ in range(num_send)]
    runs = [text for text, _ in formatted]
    cmp = compare_responses(runs)
    best = cmp["best_index"]
    return {"final_response": runs[best], "responses": [[text, sim] for text, sim in zip(runs, cmp["similarities"])], "best_index": best, "distances": cmp["distances"], "similarity": cmp["similarity"], "ok": formatted[best][1]}

def multimodel(img: Image.Image, models: list[tuple[str, float]], config: str, as_json: bool, prepare=None) -> dict:
    #returns json with fields: final_response, best_model, best_index, responses ([model, text, similarity %] per model), distances, similarity, ok
    if len(models) < 2:
        raise ValueError("Comparing models needs at least 2 models")
    formatted = []
    for model_id, temperature in models:
        if prepare:
            prepare(model_id)
        formatted.append(format_output(single_send(img, model_id, temperature, config)["final_response"], as_json))
    texts = [text for text, _ in formatted]
    cmp = compare_responses(texts)
    best = cmp["best_index"]
    return {"final_response": texts[best], "best_model": models[best][0], "best_index": best, "responses": [[model_id, text, sim] for (model_id, _), text, sim in zip(models, texts, cmp["similarities"])], "distances": cmp["distances"], "similarity": cmp["similarity"], "ok": formatted[best][1]}
