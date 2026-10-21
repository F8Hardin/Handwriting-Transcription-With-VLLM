---
title: Handwriting Transcription
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
---

# Handwriting Transcription with Vision LLMs

Turn images of handwritten text into typed text using vision-capable language models served through any **OpenAI-compatible** endpoint ([LM Studio](https://lmstudio.ai), vLLM, or Amazon Bedrock).

Built for the **Nashville Civic Tech Jam**, where the goal is to cut down manual re-typing of handwritten intake forms.

**Features**

- `image_to_text()` — image bytes in, model response out (PNG, JPEG, TIFF, and anything else Pillow reads). The prompt lives in a YAML config, so the same function does plain transcription or structured form extraction (JSON)
- Optional preprocessing — grayscale, contrast normalization, binarization, horizontal line removal
- Typed errors — failures raise specific exceptions that carry a user-facing message ("check your connection" vs "contact the administrator")
- Benchmark harness — run several models on the CVL, SMHD or FUNSD datasets; CER / WER scoring for transcription, a JSON-format check for forms
- Gradio demo — two sections (plain handwriting and forms), each with webcam or upload, plus multi-file batch upload with a live progress bar and expandable per-file results

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Test data](#test-data)
- [Project structure](#project-structure)

---

## Requirements

- Python **3.10+**
- An OpenAI-compatible vision model endpoint, e.g. LM Studio with a vision model downloaded
  (tested: `google/gemma-4-e4b`, `google/gemma-4-12b`, `google/gemma-3-12b`, `qwen/qwen3.5-9b`)

## Installation

### Use the module only

```bash
pip install git+https://github.com/F8Hardin/Handwriting-Transcription-With-VLLM.git
```

### Develop, run tests, or run the demo

```bash
git clone https://github.com/F8Hardin/Handwriting-Transcription-With-VLLM.git
cd Handwriting-Transcription-With-VLLM
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[test]"        # package + jiwer for scoring
pip install gradio              # only needed for the demo
```

## Configuration

`image_to_text()` reads its model, endpoint and prompt from a YAML file in `src/`. Three are included:

| File | Purpose |
|---|---|
| `config.yaml` | **Default.** Used when no config is given, and by the benchmark harness |
| `llm_transcription_config.yaml` | Example for plain handwriting transcription (used by the demo's transcription section) |
| `llm_forms_config.yaml` | Example for form extraction: returns JSON split into `handwritten_section` and `typed_section` (used by the demo's forms section) |

To change the default, copy an example over `config.yaml`. To use a different one for a single call, pass `config_override="llm_forms_config.yaml"`.

Every config has the same shape:

```yaml
model:
    id: google/gemma-4-12b                                # model to use
    chat_url: http://localhost:1234/v1/chat/completions   # your endpoint
    temperature: 1.0
    timeout: 300                                          # seconds

preprocess: false    # true = grayscale, contrast, binarize, remove lines

prompt: |
    You are a handwriting transcription tool.
    - Transcribe ALL handwritten text exactly as written.
    - Return ONLY the transcription text. No explanations.
```

**`src/models.yaml`** — models compared by the benchmark and listed in the demo (each with its default temperature), plus the LM Studio base URL and context length used when loading them.

> [!NOTE]
> Thinking models (Gemma 4 12B, Qwen 3.5) can spend thousands of tokens reasoning before answering. Load them with at least **16k context** or responses may come back empty.

## Usage

### As a module

```python
from handwriting_transcription import image_to_text, TranscriptionError
with open("page.png", "rb") as f:
    result = image_to_text(f.read())                          # uses config.yaml
print(result["choices"][0]["message"]["content"])
```

Override the model, temperature or config per call:

```python
result = image_to_text(image_bytes, "google/gemma-3-12b", 0.2)
result = image_to_text(image_bytes, config_override="llm_forms_config.yaml")
```

#### Errors

Every failure raises a subclass of `TranscriptionError`. `str(e)` is the technical detail (for logs); `e.user_msg` is safe to show to users.

| Error | Raised when |
|---|---|
| `ConfigError` | The config is missing or invalid, or the endpoint rejects the key or model (HTTP 401 / 403 / 404) |
| `ImageError` | The image can't be decoded or preprocessed, or the model rejects it (HTTP 400) |
| `ConnectionFailure` | The endpoint is unreachable or the request timed out |
| `ServiceError` | Rate limit (429), server error (5xx), unexpected response, or output cut off (`finish_reason: length`) |
| `EmptyResponse` | The model returned no text |

The subclasses are importable from `handwriting_transcription.llm_transcription`.

### Benchmark

Requires one of the datasets below — see [Test data](#test-data). From the project root:

```bash
python -m src.test_llm_transcription 10 42 cvl      # 10 random pages, seed 42, CVL dataset
```

The last argument is the dataset: `cvl`, `smhd` or `funsd`. The harness always uses `config.yaml`, so copy the matching example into it first (the transcription prompt for CVL / SMHD, the forms prompt for FUNSD).

Scoring depends on the dataset:

- **CVL / SMHD** — CER and WER against the ground-truth text.
- **FUNSD** — a JSON-format check only (FUNSD doesn't label which text is handwritten and which is typed): `0` = valid JSON with the expected fields, `1` = JSON but missing fields, `2` = not JSON.

Each run is saved to `runs/<timestamp>/`:

| Path | Contents |
|---|---|
| `images/` | Sampled page images (`img001_<id>.tif`, …) |
| `targets/` | Ground-truth text for CVL / SMHD (`trgt001_<id>.txt`, …) |
| `outputs/<model>/` | Model responses (`out001_<id>.txt`, …) and `error.txt` if any failed |
| `scores.csv` | CER / WER per model and image, or `json_format_error` for FUNSD |
| `seed.txt` | Seed used — rerun with the same seed to test the same pages |

`llm_analysis.ipynb` charts a run (or compares two), including a pairwise similarity table of the models' JSON outputs for form runs.

### Run the demo

```bash
python demo.py
```

Opens a Gradio page on port **1111** with a shared model selector and temperature, then two sections:

- **Handwriting transcription** — uses `llm_transcription_config.yaml`
- **Form transcription** — uses `llm_forms_config.yaml`, shows the result as formatted JSON

Each section takes a webcam photo or single upload, and also an **Upload files** button for batches: a progress bar tracks the batch (adding more files mid-run raises the total), and each file gets a green or red expandable row with the image beside its result. The demo needs LM Studio running at the `chat_url` in `src/models.yaml`; it loads and unloads models through LM Studio's API when you switch. Errors are shown to the user as short messages with a reference code, and the full detail is written to `demo_errors.log`.

With `share=True` (set in `demo.py`) it also prints a temporary public `*.gradio.live` link.

## Test data

### CVL - Handwriting
The CVL Database is **not included** — it is licensed for non-commercial research only ([CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/)).

1. Download `cvl-database-1-1.zip` and `cvl-database-cropped-1-1.zip` from [Zenodo](https://zenodo.org/records/1492267).
2. Extract both into `data/`:

   ```
   data/
   ├── cvl-database-1-1/          # trainset/ and testset/ with XML ground truth
   └── cvl-database-cropped-1-1/  # handwriting-only page images
   ```

3. Run a benchmark: `python -m src.test_llm_transcription <num_images> <seed> cvl`

If you use the CVL data, please cite:

> F. Kleber, S. Fiel, M. Diem and R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, ICDAR 2013, pp. 560–564.

### SMHD - Handwriting
The SMHD Database is **not included**. ([Find it here](https://research-repository.rmit.edu.au/articles/dataset/_b_Student_Messy_Handwritten_Dataset_SMHD_b_/24312715)).

1. Download the SMHD-forms zip file from the webpage.
2. Extract into `data/`:

   ```
   data/
   └── SMHD-forms/SMHD-forms/
   ```

3. Run a benchmark: `python -m src.test_llm_transcription <num_images> <seed> smhd`

### FUNSD - Forms
The FUNSD Database is **not included**. ([Find it here](https://guillaumejaume.github.io/FUNSD/download/)).

1. Download the FUNSD zip file from the webpage.
2. Extract into `data/`:

   ```
   data/
   └── funsd/
   ```

3. Run a benchmark: `python -m src.test_llm_transcription <num_images> <seed> funsd`

## Project structure

```
├── demo.py                            # Gradio demo (transcription + forms)
├── llm_analysis.ipynb                 # charts / comparison of benchmark runs
├── pyproject.toml
└── src/                               # installs as `handwriting_transcription`
    ├── llm_transcription.py           # image_to_text(), TranscriptionError + subclasses
    ├── config.yaml                    # default config (copy an example over it)
    ├── llm_transcription_config.yaml  # example: plain transcription
    ├── llm_forms_config.yaml          # example: form extraction (JSON)
    ├── preprocessing.py               # grayscale, contrast, binarize, line removal
    ├── models.yaml                    # models for tests / demo
    ├── test_llm_transcription.py      # benchmark harness (cvl / smhd / funsd)
    ├── alignment_test.py              # align_score(): CER / WER via jiwer; json_align_score(): format check
    └── TextExtractor.py               # extracts ground-truth text from CVL XML and SMHD files
```
