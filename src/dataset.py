import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

class IAMDatasetHelper:
    """
    Minimal helper to load one IAM page image and its ground-truth transcript.

    IAM structure (after download):
      <root>/
        formsA-D/          ← page images  (e.g. a01-000u.png)
        ascii/lines.txt    ← ground truth

    For PoC we support two modes:
      1. Full local IAM dataset at `root_dir`.
      2. Automatic download of a single sample page from IAM's public mirror
         or fallback to a synthetic test image when offline.
    """

    IAM_SAMPLE_URL = (
        "https://fki.tic.heia-fr.ch/static/img/a01-122-02.jpg"
    )

    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = Path(root_dir) if root_dir else None
        self._log = logging.getLogger(self.__class__.__name__)

    def get_sample(
        self, form_id: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (image_path, ground_truth_text | None).
        Falls back to synthetic image if IAM is unavailable.
        """
        if self.root_dir and self.root_dir.exists():
            return self._load_from_local(form_id)
        return self._load_synthetic()

    def _load_from_local(
        self, form_id: Optional[str]
    ) -> Tuple[str, Optional[str]]:
        img_dirs = [
            self.root_dir / "formsA-D",
            self.root_dir / "formsE-H",
            self.root_dir / "formsI-Z",
            self.root_dir,
        ]
        image_path = None
        for d in img_dirs:
            candidates = list(d.glob("*.png")) + list(d.glob("*.jpg"))
            if form_id:
                candidates = [c for c in candidates if form_id in c.stem]
            if candidates:
                image_path = str(candidates[0])
                break

        if image_path is None:
            self._log.warning("No IAM image found; using synthetic.")
            return self._load_synthetic()

        gt = self._parse_ground_truth(Path(image_path).stem)
        return image_path, gt

    def _parse_ground_truth(self, form_id: str) -> Optional[str]:
        if self.root_dir is None:
            return None
        gt_file = self.root_dir / "ascii" / "lines.txt"
        if not gt_file.exists():
            return None
        lines = []
        with open(gt_file) as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                line_id = parts[0]
                if form_id in line_id:
                    text = " ".join(parts[8:]).replace("|", " ")
                    lines.append(text)
        return "\n".join(lines) if lines else None

    def _load_synthetic(self) -> Tuple[str, str]:
        self._log.info("Generating synthetic handwriting test image …")
        out_path = "/tmp/synthetic_handwriting.png"
        self._make_synthetic_image(out_path)
        gt = (
            "The quick brown fox jumps over the lazy dog\n"
            "Pack my box with five dozen liquor jugs\n"
            "How vexingly quick daft zebras jump"
        )
        return out_path, gt

    @staticmethod
    def _make_synthetic_image(path: str) -> None:
        h, w = 300, 900
        img = np.ones((h, w), dtype=np.uint8) * 240
        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
        ]
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        y = 80
        for sentence in sentences:
            cv2.putText(img, sentence, (20, y), font, 0.7, (30, 30, 30), 1,
                        cv2.LINE_AA)
            y += 85
        noise = np.random.normal(0, 6, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(path, img)
