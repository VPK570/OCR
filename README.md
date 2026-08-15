# RxScanner — Handwritten Prescription OCR

End-to-end pipeline for digitizing handwritten prescriptions using deep learning. Detects text regions with CRAFT, recognizes handwriting with TrOCR (or HTR-VT), expands medical abbreviations, and exports structured results.

## Quick Start

```bash
# 1. Create conda environment
conda env create -f environment.yml
conda activate ocr_pipeline

# 2. Download CRAFT weights
# Place craft_mlt_25k.pth in CRAFT-pytorch/weights/

# 3. Run the pipeline
python pipeline/main.py --image <path-to-prescription>
python pipeline/main.py --image <path> --recognizer TrOCR --device cpu

# 4. Or launch the web UI
streamlit run frontend/app.py
```

## Architecture

```
Input Image
    │
    ▼
┌─────────────────────┐
│ Stage 1: Preprocess  │  Grayscale, denoise, CLAHE
├─────────────────────┤
│ Stage 2: Detection   │  CRAFT → word-level bounding boxes
├─────────────────────┤
│ Stage 3: Filtering   │  Area filter + NMS
├─────────────────────┤
│ Stage 4: Clustering  │  Vertical proximity → line clusters
├─────────────────────┤
│ Stage 5: Cropping    │  Extract line images with padding
├─────────────────────┤
│ Stage 6: Recognition  │  TrOCR (primary) or HTR-VT (backup)
├─────────────────────┤
│ Stage 7: Scoring     │  Confidence per line
├─────────────────────┤
│ Stage 8: Assembly    │  Reconstruct full text
├─────────────────────┤
│ Stage 9: Output      │  JSON/TXT + medical abbreviation expansion
└─────────────────────┘
    │
    ▼
Structured JSON + Plain Text
```

## Components

| Module | Description |
|--------|-------------|
| `pipeline/preprocess.py` | Grayscale conversion, denoising, CLAHE enhancement |
| `pipeline/detect.py` | CRAFT text detection with configurable thresholds |
| `pipeline/grouping.py` | Box filtering (area, NMS) and line clustering |
| `pipeline/crop.py` | Line crop extraction with padding |
| `pipeline/recognizer/trocr.py` | TrOCR (HuggingFace transformer) |
| `pipeline/recognizer/htrvt.py` | HTR-VT (ViT + CTC, requires separate env) |
| `pipeline/postprocess.py` | Confidence scoring, text reconstruction, output formatting |
| `pipeline/medical/abbreviations.py` | 100+ medical abbreviation expansions |
| `frontend/app.py` | Streamlit web interface |
| `frontend/pipeline_wrapper.py` | Bridge between frontend and pipeline |

## Dependencies

Two conda environments are required:

### Main Pipeline (`ocr_pipeline`)
- Python 3.10, PyTorch 2.4+, CUDA optional
- Install: `conda env create -f environment.yml`

### HTR-VT Recognition (`htr`)
- Python 3.8, PyTorch 1.13 (older, pinned for compat)
- Install: `cd htrvt && conda env create -f environment.yaml && cd ..`

## Required Weights

| Model | Path | Size |
|-------|------|------|
| CRAFT | `CRAFT-pytorch/weights/craft_mlt_25k.pth` | ~83 MB |
| HTR-VT | `weights/best_CER.pth` | ~50 MB |

TrOCR downloads automatically from HuggingFace on first use (~1.2 GB).

## Usage

### CLI
```bash
python pipeline/main.py --image <path>
python pipeline/main.py --image <path> --recognizer HTRVT --device cpu
python pipeline/main.py --image <path> --recognizer TrOCR --output ./my_output
```

### Frontend
```bash
streamlit run frontend/app.py
```
Features: drag-and-drop upload, confidence visualization, abbreviation expansion, JSON/TXT/CSV export, scan history.

### Demo
```bash
python demo.py                     # Show abbreviation expansion samples
python demo.py --frontend           # Launch web UI
python demo.py --image <path>       # Process an image
```

## Output Structure

```
output/
├── 01_preprocessed/
├── 02_detection_raw/
├── 03_detection_visualized/
├── 04_grouped_lines_visualized/
├── 05_line_crops/
├── 06_recognition_raw/
├── 07_recognition_with_confidence/
├── 08_final_text/
└── 09_error_analysis/
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=. --cov-report=html
```
137 tests covering abbreviation expansion, export functions, pipeline wrapper, and utilities.

## Project Structure

```
├── pipeline/            # Core OCR pipeline (9 stages)
│   ├── main.py          # Orchestrator + CLI
│   ├── preprocess.py
│   ├── detect.py
│   ├── grouping.py
│   ├── crop.py
│   ├── postprocess.py
│   ├── recognizer/      # TrOCR, HTR-VT, base class
│   ├── medical/         # Abbreviation dictionary
│   └── utils.py
├── frontend/            # Streamlit web app
│   ├── app.py
│   ├── config.py
│   ├── utils.py
│   └── pipeline_wrapper.py
├── tests/               # 137 tests
├── CRAFT-pytorch/       # Detection model (external)
├── htrvt/               # Recognition model (external, separate env)
├── samples/             # Test prescription images (add your own)
├── output/              # Pipeline output artifacts
├── environment.yml      # Conda dependencies
└── requirements.txt     # Pip dependencies
```

## Known Issues

- macOS OpenMP conflict: set `KMP_DUPLICATE_LIB_OK=TRUE`, `TOKENIZERS_PARALLELISM=false`, `OMP_NUM_THREADS=1` before imports
- HTR-VT requires its own conda env (`htr`) with PyTorch 1.13
- TrOCR downloads ~1.2 GB on first run
- `expand_text()` detects abbreviations but does not replace them in output (see tests)
