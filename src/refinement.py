import logging
import torch
import torch.nn.functional as F
from typing import List, Optional
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
from data_types import OCRResult

class TextRefiner:
    """
    Uses DistilBERT masked language model to probabilistically correct
    low-confidence tokens in the OCR output.
    Strategy:
      - Tokenize each word.
      - Mask it one at a time.
      - Accept the MLM top-1 prediction if its probability exceeds the
        original token probability by `replacement_margin`.
    No rule-based or regex-based corrections — purely probabilistic.
    """

    MODEL_ID = "distilbert-base-uncased"
    MASK_TOKEN = "[MASK]"

    def __init__(
        self,
        device: Optional[str] = None,
        replacement_margin: float = 0.25,
        min_word_length: int = 2,
        max_context_tokens: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.replacement_margin = replacement_margin
        self.min_word_length = min_word_length
        self.max_context_tokens = max_context_tokens
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.info(f"Loading DistilBERT ({self.MODEL_ID}) …")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.MODEL_ID)
        self.model = DistilBertForMaskedLM.from_pretrained(self.MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self._log.info("  DistilBERT ready.")

    @torch.no_grad()
    def refine(self, ocr_results: List[OCRResult]) -> str:
        refined_lines = []
        for result in ocr_results:
            refined = self._refine_line(result.raw_text, result.confidence)
            refined_lines.append(refined)
            if refined != result.raw_text:
                self._log.info(
                    f"  Line {result.line_index:03d} refined:\n"
                    f"    BEFORE: {result.raw_text!r}\n"
                    f"    AFTER : {refined!r}"
                )
        return "\n".join(refined_lines)

    def _refine_line(self, text: str, line_confidence: float) -> str:
        words = text.split()
        if not words:
            return text

        refined_words = list(words)
        for word_idx, word in enumerate(words):
            if len(word) < self.min_word_length:
                continue
            refined_words[word_idx] = self._evaluate_word(
                words, word_idx, line_confidence
            )

        return " ".join(refined_words)

    def _evaluate_word(
        self, words: List[str], target_idx: int, line_conf: float
    ) -> str:
        original_word = words[target_idx]
        context_start = max(0, target_idx - self.max_context_tokens // 2)
        context_end = min(len(words), target_idx + self.max_context_tokens // 2)

        context_words = list(words[context_start:context_end])
        local_idx = target_idx - context_start

        masked_words = (
            context_words[:local_idx]
            + [self.MASK_TOKEN]
            + context_words[local_idx + 1:]
        )
        masked_text = " ".join(masked_words)
        original_text = " ".join(context_words)

        enc_masked = self.tokenizer(
            masked_text, return_tensors="pt", truncation=True
        ).to(self.device)
        enc_original = self.tokenizer(
            original_text, return_tensors="pt", truncation=True
        ).to(self.device)

        mask_positions = (
            enc_masked.input_ids[0] == self.tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            return original_word

        mask_pos = mask_positions[0].item()

        logits_masked = self.model(**enc_masked).logits[0, mask_pos]
        probs_masked = F.softmax(logits_masked, dim=-1)

        top_id = probs_masked.argmax().item()
        top_token = self.tokenizer.convert_ids_to_tokens([top_id])[0]
        top_prob = probs_masked[top_id].item()

        orig_ids = self.tokenizer(
            original_word, add_special_tokens=False
        ).input_ids
        if not orig_ids:
            return original_word

        orig_id = orig_ids[0]
        orig_token_logits = self.model(**enc_original).logits[0]
        orig_token_pos = self._find_token_position(
            enc_original.input_ids[0], orig_id
        )
        if orig_token_pos is None:
            orig_prob = 0.0
        else:
            orig_prob = (
                F.softmax(orig_token_logits[orig_token_pos], dim=-1)[orig_id]
                .item()
            )

        self._log.debug(
            f"    Word '{original_word}': orig_prob={orig_prob:.3f}, "
            f"top_pred='{top_token}' ({top_prob:.3f})"
        )

        if (
            top_prob - orig_prob > self.replacement_margin
            and not top_token.startswith("##")
            and top_token not in ("[UNK]", "[PAD]", "[CLS]", "[SEP]")
        ):
            return top_token

        return original_word

    @staticmethod
    def _find_token_position(
        input_ids: torch.Tensor, token_id: int
    ) -> Optional[int]:
        positions = (input_ids == token_id).nonzero(as_tuple=True)[0]
        return positions[0].item() if len(positions) > 0 else None
