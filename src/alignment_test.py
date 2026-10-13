# Score a transcription against its target text (CER / WER).
# CER - character error rate
# WER - word error rate

import re
import jiwer

def normalize(text: str) -> str:
    # Make output comparable to CVL targets (words only, no punctuation).
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def align_score(target: str, prediction: str, verbose: bool = False) -> dict:
    # Compare one transcription to its target.
    # Returns: {"cer": float, "wer": float, "alignment": str}
    # cer/wer: 0.0 = perfect. None if the target is empty (nothing to score).
    # alignment: REF/HYP word view with S/D/I markers under each mistake.
    ref, hyp = normalize(target), normalize(prediction)
    if not ref:                                    # blank CVL page
        return {"cer": None, "wer": None, "alignment": ""}

    alignment = jiwer.visualize_alignment(jiwer.process_words(ref, hyp), show_measures=False)
    result = {
        "cer": jiwer.cer(ref, hyp),
        "wer": jiwer.wer(ref, hyp),
        "alignment": alignment,
    }
    if verbose:
        print(f"CER {result['cer']:.1%}  WER {result['wer']:.1%}")
        print(alignment)
    return result
