"""
recognizer/htrvt.py — HTR-VT recognizer

Imports the HTR-VT model directly from the htrvt/model/ repo.
Uses CTC decoding + greedy best-path.

Weights: best_CER.pth or best_WER.pth
"""

import sys
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import OrderedDict
from typing import Optional, Tuple, List

# Add pipeline root to import base
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from recognizer.base import BaseRecognizer
from utils import get_logger


class CTCLabelConverter:
    """
    Greedy CTC decoder.
    Converts raw token-index tensors to text strings.
    """

    # Alphabet matching the best_CER.pth / best_WER.pth checkpoints
    ALPHABET = (
        " ()+,-./0123456789:<>ABCDEFGHIJKLMNOPQRSTUVWYZ[]"
        "abcdefghijklmnopqrstuvwxyz¾Ößäöüÿāēōūȳ̄̈—"
    )

    def __init__(self, character: str = None):
        character = character or self.ALPHABET
        self.dict = {char: i + 1 for i, char in enumerate(character)}
        if len(self.dict) == 87:
            self.dict["["] = 88
            self.dict["]"] = 89
        self.character = ["[blank]"] + list(character)
        if len(self.character) == 88:
            self.character.extend(["[", "]"])

    def decode(self, text_index: torch.Tensor, length: torch.Tensor) -> List[str]:
        """Best-path (greedy) CTC decode."""
        texts = []
        idx = 0
        for l in length:
            t = text_index[idx: idx + l]
            chars = []
            for i in range(l):
                if (
                    t[i] != 0
                    and (not (i > 0 and t[i - 1] == t[i]))
                    and t[i] < len(self.character)
                ):
                    chars.append(self.character[t[i]])
            texts.append("".join(chars))
            idx += l
        return texts


class HTRVTRecognizer(BaseRecognizer):
    """
    Wraps the HTR-VT model (Vision Transformer + CTC).

    Args:
        model_path  : Path to best_CER.pth or best_WER.pth.
        htrvt_dir   : Path to the htrvt/ repo root (for sys.path injection).
        device      : 'cpu', 'cuda', or None for auto.
        img_size    : (width, height) fed into model. Default (512, 64).
    """

    def __init__(
        self,
        model_path: str,
        htrvt_dir: str,
        device: Optional[str] = None,
        img_size: Tuple[int, int] = (512, 64),
    ):
        self._log = get_logger("HTRVTRecognizer")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size   # (W, H)

        # ── Inject htrvt ROOT dir into sys.path ──
        # HTR_VT.py does `from model import resnet18`, so the parent of model/
        # (i.e. htrvt/) must be on sys.path, NOT htrvt/model/
        htrvt_root = os.path.abspath(htrvt_dir)
        if htrvt_root not in sys.path:
            sys.path.insert(0, htrvt_root)

        from model import HTR_VT
        self._htrvt = HTR_VT

        # ── Build model architecture ──
        # nb_cls = 80  (standard IAM classes)
        self._log.info(f"Building HTR-VT model on {self.device} ...")
        self.model = HTR_VT.create_model(
            nb_cls=80,
            img_size=[img_size[1], img_size[0]]   # [H, W] as expected by HTR-VT
        )

        # ── Load weights ──
        self._log.info(f"Loading weights from {model_path} ...")
        ckpt = torch.load(model_path, map_location="cpu")
        # Handle state_dict keys (EMA or plain)
        state_dict = ckpt.get("state_dict_ema", ckpt.get("state_dict", ckpt))
        # Strip 'module.' prefix if present (DataParallel)
        clean = OrderedDict()
        for k, v in state_dict.items():
            clean[re.sub(r"^module\.", "", k)] = v
        self.model.load_state_dict(clean, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.converter = CTCLabelConverter()
        self._log.info("  HTR-VT ready.")

    def preprocess(self, image_crop: np.ndarray) -> torch.Tensor:
        """
        Converts a BGR or grayscale crop to a normalised tensor.
        Resizes to (W=512, H=64) as expected by the checkpoint.
        Returns: [1, 1, H, W] float32 tensor in [0, 1].
        """
        import cv2
        W, H = self.img_size
        # Grayscale
        if len(image_crop.shape) == 3:
            gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_crop
        # Resize
        pil = Image.fromarray(gray).convert("L").resize((W, H), Image.BILINEAR)
        tensor = torch.from_numpy(np.array(pil)).float() / 255.0
        return tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # [1,1,H,W]

    @torch.no_grad()
    def predict(self, image_crop: np.ndarray) -> Tuple[str, float]:
        """
        Recognizes text in a single line crop.

        Returns:
            (text, confidence)
        """
        x = self.preprocess(image_crop)       # [1, 1, H, W]
        preds = self.model(x)                  # [1, T, nb_cls]
        preds = preds.float()

        preds_t = preds.permute(1, 0, 2).log_softmax(2)  # [T, 1, nb_cls]
        preds_size = torch.IntTensor([preds.size(1)])

        _, preds_index = preds_t.max(2)
        preds_index = preds_index.transpose(1, 0).contiguous().view(-1)

        text = self.converter.decode(preds_index.data, preds_size.data)[0]

        # Confidence: mean max softmax probability across sequence
        max_probs = preds.softmax(-1).max(-1).values  # [1, T]
        confidence = float(max_probs.mean().clamp(0.0, 1.0).item())

        self._log.debug(f"    → '{text}'  (conf={confidence:.3f})")
        return text, confidence
