# Handwriting Transcription with Vision LLMs

Turn images of handwritten text into typed text using vision-capable language models served through any **OpenAI-compatible** endpoint ([LM Studio](https://lmstudio.ai), vLLM, or Amazon Bedrock).

Built for the **Nashville Civic Tech Jam**, where the goal is to cut down manual re-typing of handwritten intake forms.

**Features**

- 📄 `image_to_text()` — image bytes in, transcription out (PNG, JPEG, TIFF, and anything else Pillow reads)
- 🔧 Optional preprocessing — grayscale, contrast normalization, binarization, horizontal line removal
- 📊 Benchmark harness — run several models on the [CVL handwriting database](https://cvl.tuwien.ac.at/research/cvl-databases/an-off-line-database-for-writer-retrieval-writer-identification-and-word-spotting/) and score character / word error rate (CER / WER)
- 🎥 Gradio demo — webcam or upload, pick a model, transcribe live

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Test data (CVL)](#test-data-cvl)
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

**`src/llm_transcription_config.yaml`** — default model and endpoint used by `image_to_text()`:

```yaml
model:
    id: google/gemma-4-e4b                                # model to use
    chat_url: http://localhost:1234/v1/chat/completions   # your endpoint
    temperature: 1.0
    timeout: 300                                          # seconds

preprocess: false    # true = grayscale, contrast, binarize, remove lines
```

**`src/models.yaml`** — models compared by the test harness and listed in the demo, plus the LM Studio base URL and context length used when loading them.

> [!NOTE]
> Thinking models (Gemma 4 12B, Qwen 3.5) can spend thousands of tokens reasoning before answering. Load them with at least **16k context** or responses may come back empty.

## Usage

### As a module

```python
from handwriting_transcription import image_to_text, TranscriptionError
with open("page.png", "rb") as f:
    result = image_to_text(f.read())                          # uses the config's model
print(result["choices"][0]["message"]["content"])
```

Override the model or temperature per call:

```python
result = image_to_text(image_bytes, "google/gemma-3-12b", 0.2)
```

`TranscriptionError` is raised when the request fails (unreachable endpoint, HTTP error, etc.).

### Benchmark on CVL data

Requires the CVL data — see [Test data (CVL)](#test-data-cvl). From the project root:

```bash
python -m src.test_llm_transcription 10 42      # 10 random pages, seed 42
```

Each run is saved to `runs/<timestamp>/`:

| Path | Contents |
|---|---|
| `images/` | Sampled page images (`img001_<cvl-id>.tif`, …) |
| `targets/` | Ground-truth text (`trgt001_<cvl-id>.txt`, …) |
| `outputs/<model>/` | Model transcriptions (`out001_<cvl-id>.txt`, …) and `error.txt` if any failed |
| `scores.csv` | CER / WER per model and image |
| `seed.txt` | Seed used — rerun with the same seed to test the same pages |

### Run the demo

```bash
python demo.py
```

Opens a Gradio page on port **7860** with webcam/upload, model selection and temperature. With `share=True` (set in `demo.py`) it also prints a temporary public `*.gradio.live` link.

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

3. Run a benchmark: `python -m src.test_llm_transcription <num_images> <seed> CVL`

If you use the CVL data, please cite:

> F. Kleber, S. Fiel, M. Diem and R. Sablatnig, *CVL-Database: An Off-line Database for Writer Retrieval, Writer Identification and Word Spotting*, ICDAR 2013, pp. 560–564.

### SMHD - Handwriting
The SMHD Database is **not included**. ([Find it here](https://research-repository.rmit.edu.au/articles/dataset/_b_Student_Messy_Handwritten_Dataset_SMHD_b_/24312715)).

1. Download the SMHD-forms zip file from the webpade.

2. Extract into `data/`:

   ```
   data/
   ├── SMHD-forms/SMHD-forms/
   ```

3. Run a benchmark: `python -m src.test_llm_transcription <num_images> <seed> SMHD`

### FUNSD
The FUNSD Database is **not included**. ([Find it here](https://guillaumejaume.github.io/FUNSD/download/)).

1. Download the FUNSD zip file from the webpade.

2. Extract into `data/`:

   ```
   data/
   ├── funsd/
   ```

3. Run a benchmark: `python -m src.test_llm_transcription <num_images> <seed> funsd`

## Project structure

```
├── demo.py                          # Gradio demo
├── llm_analysis.ipynb               # charts / comparison of benchmark runs
├── pyproject.toml
└── src/                             # installs as `handwriting_transcription`
    ├── llm_transcription.py         # image_to_text(), TranscriptionError
    ├── llm_transcription_config.yaml
    ├── preprocessing.py             # grayscale, contrast, binarize, line removal
    ├── models.yaml                  # models for tests / demo
    ├── test_llm_transcription.py    # benchmark harness
    ├── alignment_test.py            # align_score(): CER / WER via jiwer
    ├── CVL_XML_TextExtractor.py     # extracts ground truth from CVL XML
    └── test_CVL_XML_TextExtractor.sh
```
