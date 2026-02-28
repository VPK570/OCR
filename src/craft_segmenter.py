import logging
import cv2
import numpy as np
import easyocr
from typing import List, Tuple
from data_types import LineRegion

class CraftLineSegmenter:
    """
    Uses CRAFT (via EasyOCR) to detect word-level bounding boxes
    and groups them into cohesive lines.
    """

    def __init__(
        self,
        gpu: bool = True,
        overlap_threshold: float = 0.5,
        padding: int = 4
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.info("Initializing CRAFT segmenter (EasyOCR) ...")
        self.reader = easyocr.Reader(['en'], gpu=gpu)
        self.overlap_threshold = overlap_threshold
        self.padding = padding

    def segment(self, image_path: str, original_gray: np.ndarray) -> List[LineRegion]:
        """
        Detects text regions and groups them into lines.
        """
        self._log.info(f"Detecting text in {image_path} ...")
        # EasyOCR's detect() returns word-level boxes
        # Format: [ [x, y, w, h], ... ]
        horizontal_list, free_list = self.reader.detect(image_path)
        
        # Combine both lists of boxes
        boxes = []
        if horizontal_list and len(horizontal_list) > 0:
            for box in horizontal_list[0]:
                xmin, xmax, ymin, ymax = box
                boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))
            
        if free_list and len(free_list) > 0:
            for poly in free_list[0]:
                poly = np.array(poly)
                xmin, ymin = poly.min(axis=0)
                xmax, ymax = poly.max(axis=0)
                boxes.append((int(xmin), int(ymin), int(xmax), int(ymax)))

        if not boxes:
            self._log.warning("No text detected by CRAFT.")
            return []

        # Group boxes into lines
        lines = self._group_into_lines(boxes)
        self._log.info(f"Grouped {len(boxes)} boxes into {len(lines)} lines.")

        h, w = original_gray.shape
        regions: List[LineRegion] = []
        for idx, line_box in enumerate(lines):
            xmin, ymin, xmax, ymax = line_box
            
            # Apply padding
            y0 = max(0, ymin - self.padding)
            y1 = min(h, ymax + self.padding)
            x0 = max(0, xmin - self.padding)
            x1 = min(w, xmax + self.padding)

            crop = original_gray[y0:y1, x0:x1]
            regions.append(
                LineRegion(
                    y_start=y0, y_end=y1,
                    x_start=x0, x_end=x1,
                    image_crop=crop
                )
            )
            self._log.debug(f"  Line {idx:03d}: y=[{y0},{y1}] x=[{x0},{x1}]")

        return regions

    def _group_into_lines(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Groups word boxes into lines based on vertical overlap.
        """
        # Sort by vertical center
        boxes = sorted(boxes, key=lambda b: (b[1] + b[3]) / 2)
        
        lines = []
        if not boxes:
            return lines

        current_line = boxes[0]
        
        for next_box in boxes[1:]:
            # Check vertical overlap
            c_y_min, c_y_max = current_line[1], current_line[3]
            n_y_min, n_y_max = next_box[1], next_box[3]
            
            overlap = max(0, min(c_y_max, n_y_max) - max(c_y_min, n_y_min))
            min_h = min(c_y_max - c_y_min, n_y_max - n_y_min)
            
            if overlap / min_h > self.overlap_threshold:
                # Merge into current line
                current_line = (
                    min(current_line[0], next_box[0]),
                    min(current_line[1], next_box[1]),
                    max(current_line[2], next_box[2]),
                    max(current_line[3], next_box[3])
                )
            else:
                # New line
                lines.append(current_line)
                current_line = next_box
        
        lines.append(current_line)
        
        # Finally, sort lines from top to bottom
        lines = sorted(lines, key=lambda l: l[1])
        return lines
