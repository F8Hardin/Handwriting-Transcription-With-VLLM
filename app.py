import sys
import runpy
from pathlib import Path
sys.argv = [sys.argv[0], "models_gemini.yaml", "--port", "7860"]    # Hugging Face Spaces serves port 7860
runpy.run_path(str(Path(__file__).parent / "demo.py"), run_name="__main__")
