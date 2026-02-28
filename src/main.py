import argparse
import logging
import os
from typing import Optional

from dataset import IAMDatasetHelper
from pipeline import HandwritingDigitizationPipeline
from preprocessing import Preprocessor
from segmentation import LineSegmenter
from craft_segmenter import CraftLineSegmenter
from ocr import OCRModel
from htrvt_wrapper import HTRVTModel
from refinement import TextRefiner
from evaluation import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("htd_pipeline")

def build_pipeline(
    device: Optional[str] = None, 
    ocr_type: str = "trocr",
    segmenter_type: str = "standard",
    htrvt_path: Optional[str] = None
) -> HandwritingDigitizationPipeline:
    
    preprocessor = Preprocessor(
        blur_kernel=(3, 3),
        block_size=25,
        c_constant=10,
        deskew=True,
    )
    
    if segmenter_type == "craft":
        segmenter = CraftLineSegmenter(gpu=(device != "cpu"), padding=4)
    else:
        segmenter = LineSegmenter(
            min_line_height_ratio=0.01,
            valley_threshold_percentile=15.0,
            dilation_iterations=2,
            padding=4,
        )
    
    if ocr_type == "htrvt":
        model_path = htrvt_path or "../HTR-VT/data/read/best_CER.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"HTR-VT model not found at {model_path}. Please provide valid --htrvt-path")
        ocr = HTRVTModel(model_path=model_path, device=device)
    else:
        ocr = OCRModel(device=device, batch_size=4)
        
    refiner = TextRefiner(
        device=device,
        replacement_margin=0.25,
        min_word_length=2,
        max_context_tokens=64,
    )
    evaluator = Evaluator()

    return HandwritingDigitizationPipeline(
        preprocessor=preprocessor,
        segmenter=segmenter,
        ocr=ocr,
        refiner=refiner,
        evaluator=evaluator,
    )

def main():
    parser = argparse.ArgumentParser(
        description="Handwritten Text Digitization Pipeline"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to a handwritten page image. If omitted, uses IAM sample."
    )
    parser.add_argument(
        "--iam-root", type=str, default=None,
        help="Path to local IAM dataset root directory."
    )
    parser.add_argument(
        "--form-id", type=str, default=None,
        help="IAM form ID to select (e.g. 'a01-000u')."
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Torch device override (e.g. 'cpu', 'cuda:0')."
    )
    parser.add_argument(
        "--ground-truth", type=str, default=None,
        help="Path to a .txt file with ground truth for evaluation."
    )
    parser.add_argument(
        "--model", type=str, choices=["trocr", "htrvt"], default="trocr",
        help="OCR model engine to use."
    )
    parser.add_argument(
        "--segmenter", type=str, choices=["standard", "craft"], default="standard",
        help="Segmentation algorithm to use."
    )
    parser.add_argument(
        "--htrvt-path", type=str, default=None,
        help="Path to HTR-VT checkpoint .pth file."
    )
    
    args = parser.parse_args()

    print("\n" + "━" * 60)
    print("  Handwritten Text Digitization Pipeline")
    print(f"  Engine: {args.model.upper()} | Segmenter: {args.segmenter.upper()}")
    print("━" * 60 + "\n")

    iam = IAMDatasetHelper(root_dir=args.iam_root)

    if args.image:
        image_path = args.image
        ground_truth = None
        if args.ground_truth:
            with open(args.ground_truth) as f:
                ground_truth = f.read()
    else:
        image_path, ground_truth = iam.get_sample(form_id=args.form_id)

    print(f"  Image        : {image_path}")
    print(f"  Ground truth : {'provided' if ground_truth else 'none'}")
    print(f"  Device       : {args.device or 'auto'}\n")

    pipeline = build_pipeline(
        device=args.device,
        ocr_type=args.model,
        segmenter_type=args.segmenter,
        htrvt_path=args.htrvt_path
    )
    output = pipeline.run(image_path, ground_truth=ground_truth)

    print("\n" + "━" * 60)
    print("  FINAL RAW TEXT")
    print("━" * 60)
    print(output.raw_text)

    print("\n" + "━" * 60)
    print("  FINAL REFINED TEXT")
    print("━" * 60)
    print(output.refined_text)

    if ground_truth is not None and not (isinstance(output.cer_before, float) and np.isnan(output.cer_before)):
        print("\n" + pipeline.evaluator.full_report(output))

    print("\n  Per-line OCR output:")
    for r in output.ocr_results:
        print(f"    Line {r.line_index:03d}: {r.raw_text!r}")

    print("\n  Pipeline complete.\n")
    return output

if __name__ == "__main__":
    main()
