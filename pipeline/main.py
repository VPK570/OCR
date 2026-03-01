"""
main.py — Prescription Digitization Pipeline Entry Point

Usage:
    python main.py --image <path>
    python main.py --image <path> --recognizer TrOCR
    python main.py --image <path> --recognizer HTRVT --device cpu

All intermediate outputs are saved to output/ subdirectories.
"""

# ── macOS / OpenMP conflict fix ─────────────────────────────────────────────
# Must be set BEFORE torch or transformers are imported.
# Prevents segfault caused by PyTorch and TensorFlow sharing OpenMP in the
# same conda environment (common on Apple Silicon and x86 macOS).
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["USE_TORCH"] = "1"
# ────────────────────────────────────────────────────────────────────────────

import sys
import argparse
import logging

import cv2

# ═══════════════════════════════════════════════════
# ▶  CONFIGURATION  (edit here or pass CLI args)
# ═══════════════════════════════════════════════════

CONFIG = {
    # Recognizer to use: "HTRVT" or "TrOCR"
    "USE_RECOGNIZER": "TrOCR",

    # Confidence below which a line is flagged for review
    "CONFIDENCE_THRESHOLD": 0.75,

    # Enable debug logging and saving intermediate outputs
    "DEBUG": True,

    # Paths — relative to this file's directory
    "CRAFT_DIR":     os.path.join(os.path.dirname(__file__), "..", "CRAFT-pytorch"),
    "CRAFT_WEIGHTS": os.path.join(os.path.dirname(__file__), "..", "CRAFT-pytorch", "weights", "craft_mlt_25k.pth"),
    "HTRVT_DIR":     os.path.join(os.path.dirname(__file__), "..", "htrvt"),
    "HTRVT_WEIGHTS": os.path.join(os.path.dirname(__file__), "..", "weights", "best_CER.pth"),
    "OUTPUT_DIR":    os.path.join(os.path.dirname(__file__), "..", "output"),

    # Preprocessing options
    "USE_CLAHE": True,
    "DESKEW":    False,   # rotation disabled

    # CRAFT options
    "CUDA":            False,
    "TEXT_THRESHOLD":  0.7,
    "LINK_THRESHOLD":  0.4,
    "LOW_TEXT":        0.4,

    # Box grouping options
    "MIN_BOX_AREA":  100,
    "IOU_THRESHOLD": 0.5,

    # Crop padding (pixels)
    "CROP_PADDING": 8,
}

# ═══════════════════════════════════════════════════
# Logging setup
# ═══════════════════════════════════════════════════

logging.basicConfig(
    level=logging.DEBUG if CONFIG["DEBUG"] else logging.INFO,
    format="[%(asctime)s] %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Pipeline")

# ═══════════════════════════════════════════════════
# Add pipeline/ to sys.path so imports work
# ═══════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from utils       import setup_output_dirs
from preprocess  import Preprocessor
from detect      import CRAFTDetector
from grouping    import BoxGrouper
from crop        import LineCropper
from postprocess import TextReconstructor


# ═══════════════════════════════════════════════════
# Pipeline Builder
# ═══════════════════════════════════════════════════

def build_recognizer(cfg: dict, device: str = None):
    """Instantiates the selected recognizer from config."""
    rec_type = cfg["USE_RECOGNIZER"].upper()
    dev = device or ("cuda" if cfg["CUDA"] else "cpu")

    if rec_type == "HTRVT":
        from recognizer.htrvt import HTRVTRecognizer
        return HTRVTRecognizer(
            model_path=cfg["HTRVT_WEIGHTS"],
            htrvt_dir=cfg["HTRVT_DIR"],
            device=dev,
        )
    elif rec_type == "TROCR":
        from recognizer.trocr import TrOCRRecognizer
        return TrOCRRecognizer(device=dev)
    else:
        raise ValueError(f"Unknown recognizer: {rec_type}. Choose HTRVT or TrOCR.")


# ═══════════════════════════════════════════════════
# Main Pipeline Orchestrator
# ═══════════════════════════════════════════════════

def run(image_path: str, cfg: dict, device: str = None):
    """
    Runs all 9 stages on a single image.

    Returns:
        full_text       : Reconstructed text string.
        structured_json : Dict with per-line text + confidence.
    """
    output_dir = cfg["OUTPUT_DIR"]
    image_name = os.path.basename(image_path)

    logger.info("=" * 60)
    logger.info("  Prescription Digitization Pipeline")
    logger.info(f"  Image    : {image_path}")
    logger.info(f"  Recognizer: {cfg['USE_RECOGNIZER']}")
    logger.info(f"  Output   : {output_dir}")
    logger.info("=" * 60)

    setup_output_dirs(output_dir)

    # ── Stage 1: Preprocessing ──────────────────────
    logger.info("\n─── Stage 1: Preprocessing ───")
    preprocessor = Preprocessor(
        blur_kernel=(3, 3),
        use_clahe=cfg["USE_CLAHE"],
        deskew=cfg["DESKEW"],
        output_dir=output_dir,
    )
    gray, original_bgr = preprocessor.process(image_path)

    # ── Stage 2: CRAFT Detection ────────────────────
    logger.info("\n─── Stage 2: CRAFT Detection ───")
    detector = CRAFTDetector(
        craft_dir=cfg["CRAFT_DIR"],
        weights_path=cfg["CRAFT_WEIGHTS"],
        cuda=cfg["CUDA"],
        text_threshold=cfg["TEXT_THRESHOLD"],
        link_threshold=cfg["LINK_THRESHOLD"],
        low_text=cfg["LOW_TEXT"],
        output_dir=output_dir,
    )
    rects = detector.run(gray, image_name=image_name)

    if not rects:
        logger.warning("  No text detected. Exiting.")
        return "", {"lines": []}

    # ── Stages 3 & 4: Box Grouping ──────────────────
    logger.info("\n─── Stages 3–4: Box Filtering & Line Clustering ───")
    grouper = BoxGrouper(
        min_box_area=cfg["MIN_BOX_AREA"],
        iou_threshold=cfg["IOU_THRESHOLD"],
        output_dir=output_dir,
    )
    line_clusters = grouper.run(rects, original_bgr, image_name=image_name)

    if not line_clusters:
        logger.warning("  No line clusters found. Exiting.")
        return "", {"lines": []}

    # ── Stage 5: Line Cropping ──────────────────────
    logger.info("\n─── Stage 5: Line Cropping ───")
    cropper = LineCropper(padding=cfg["CROP_PADDING"], output_dir=output_dir)
    crops = cropper.crop_lines(original_bgr, line_clusters)

    # ── Stage 6: Recognition ────────────────────────
    logger.info(f"\n─── Stage 6: Recognition ({cfg['USE_RECOGNIZER']}) ───")
    recognizer = build_recognizer(cfg, device=device)

    line_results = []
    for line_id, crop in crops:
        logger.info(f"  Recognizing line {line_id:02d} ...")
        try:
            text, confidence = recognizer.predict(crop)
            line_results.append((line_id, text, confidence, crop))
        except Exception as e:
            logger.error(f"  FAILED line {line_id:02d}: {str(e)}")
            line_results.append((line_id, "[RECOGNITION FAILURE]", 0.0, crop))

    # ── Stages 7–9: Post-processing & Output ────────
    logger.info("\n─── Stages 7–9: Confidence Scoring & Output ───")
    reconstructor = TextReconstructor(
        confidence_threshold=cfg["CONFIDENCE_THRESHOLD"],
        output_dir=output_dir,
    )
    full_text, structured_json = reconstructor.reconstruct(line_results)

    # ── Final Summary ────────────────────────────────
    flagged = [l for l in structured_json["lines"] if l["flagged"]]
    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  Lines detected   : {len(line_clusters)}")
    logger.info(f"  Lines recognized : {len(line_results)}")
    logger.info(f"  Lines flagged    : {len(flagged)}")
    logger.info(f"  Output dir       : {output_dir}")
    logger.info("=" * 60)
    logger.info("\n  ── RECONSTRUCTED TEXT ──")
    for line in full_text.splitlines():
        logger.info(f"  {line}")

    return full_text, structured_json


# ═══════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Handwritten Prescription Digitization Pipeline"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input handwritten image."
    )
    parser.add_argument(
        "--recognizer", type=str, choices=["HTRVT", "TrOCR"],
        default=None,
        help="OCR engine. Default: from CONFIG['USE_RECOGNIZER']."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device: 'cpu' or 'cuda'. Default: auto."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory. Defaults to ../output."
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Suppress debug logging."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override config with CLI args
    cfg = dict(CONFIG)
    if args.recognizer:
        cfg["USE_RECOGNIZER"] = args.recognizer
    if args.output:
        cfg["OUTPUT_DIR"] = args.output
    if args.no_debug:
        cfg["DEBUG"] = False
        logging.getLogger().setLevel(logging.INFO)

    text, json_out = run(args.image, cfg, device=args.device)

    print("\n" + "=" * 60)
    print("  FINAL RECONSTRUCTED TEXT")
    print("=" * 60)
    print(text)
    print("=" * 60)
