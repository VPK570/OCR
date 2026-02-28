import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
from collections import OrderedDict
import re

import htrvt_model as HTR_VT
from data_types import LineRegion, OCRResult

class CTCLabelConverter:
    """
    Standard CTC label converter for HTR-VT.
    """
    def __init__(self, character: str):
        # Base alphabet
        self.dict = {char: i + 1 for i, char in enumerate(character)}
        # Special case for [ and ] found in some research datasets
        if len(self.dict) == 87:
            self.dict['['], self.dict[']'] = 88, 89
            
        self.character = ['[blank]'] + list(character)
        if len(self.character) == 88:
            self.character.extend(['[', ']'])

    def decode(self, text_index: torch.Tensor, length: torch.Tensor) -> List[str]:
        """
        Decodes CTC output using best-path (greedy).
        """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            char_list = []
            for i in range(l):
                # Greedy CTC decoding logic
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])) and t[i] < len(self.character):
                    char_list.append(self.character[t[i]])
            texts.append(''.join(char_list))
            index += l
        return texts

class HTRVTModel:
    """
    Wrapper for the HTR-VT model architecture and weights.
    Fits the modular OCR pipeline interface.
    """
    
    # Best-fit alphabet for the provided best_CER.pth checkpoint
    ALPHABET = " ()+,-./0123456789:<>ABCDEFGHIJKLMNOPQRSTUVWYZ[]abcdefghijklmnopqrstuvwxyz¾Ößäöüÿāēōūȳ̄̈—"

    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        img_size: Tuple[int, int] = (512, 64)
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size
        self._log = logging.getLogger(self.__class__.__name__)
        
        self._log.info(f"Building HTR-VT model architecture on {self.device} ...")
        # nb_cls = 90 (alphabet 87 + blank 1 + brackets 2)
        self.model = HTR_VT.create_model(nb_cls=90, img_size=img_size[::-1])
        
        self._log.info(f"Loading weights from {model_path} ...")
        ckpt = torch.load(model_path, map_location='cpu')
        
        # Cleanup state dict (remove 'module.' prefix if present)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        state_dict = ckpt.get('state_dict_ema', ckpt.get('state_dict', ckpt))
        
        for k, v in state_dict.items():
            clean_name = re.sub(pattern, '', k)
            model_dict[clean_name] = v
            
        self.model.load_state_dict(model_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()
        
        self.converter = CTCLabelConverter(self.ALPHABET)
        self._log.info("HTR-VT model is ready.")

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Converts crop to grayscale, resizes to target size, and normalizes.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # Resize as per HTR-VT expectations (fixed W, fixed H)
        # Model expects W=512, H=64 (img_size[::-1] in HTR-VT code)
        img_pil = Image.fromarray(image).convert('L')
        img_pil = img_pil.resize(self.img_size, Image.BILINEAR)
        
        img_tensor = torch.from_numpy(np.array(img_pil)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        return img_tensor.to(self.device)

    @torch.no_grad()
    def recognize_lines(self, regions: List[LineRegion]) -> List[OCRResult]:
        results = []
        for i, region in enumerate(regions):
            input_tensor = self.preprocess(region.image_crop)
            preds = self.model(input_tensor)
            
            # CTC output processing
            preds = preds.float()
            preds_size = torch.IntTensor([preds.size(1)])
            preds = preds.permute(1, 0, 2).log_softmax(2)
            
            _, preds_index = preds.max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            
            decoded_text = self.converter.decode(preds_index.data, preds_size.data)[0]
            
            # Confidence score (proxy from mean logprob of max sequence)
            # Simplified for now as HTR-VT doesn't explicitly return score in example
            conf = 1.0 # placeholder
            
            self._log.info(f"  Line {i:03d}: {decoded_text!r}")
            results.append(
                OCRResult(
                    line_index=i,
                    raw_text=decoded_text,
                    confidence=conf,
                    region=region
                )
            )
        return results
