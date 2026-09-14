"""Core detection modules: Face Recognition, YOLOv8 Object Detection,
EasyOCR text extraction, and Dynamic State Management.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import re
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

import config

try:
    import face_recognition
except ImportError:
    face_recognition = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import easyocr
except ImportError:
    easyocr = None


@dataclass
class FaceDetection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    is_rich: bool
    distance: float
    name: str


@dataclass
class CardDetection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: str
    confidence: float
    is_trigger: bool


@dataclass
class PetDetection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    species: str
    confidence: float


class FaceRecognitionManager:
    """Manages reference face loading and real-time face recognition."""

    def __init__(self, reference_image_path: Optional[Path] = None):
        self.reference_path = reference_image_path or config.REFERENCE_IMAGE_PATH
        self.reference_encoding: Optional[np.ndarray] = None
        self.tolerance = config.FACE_MATCH_TOLERANCE
        self.downscale = config.FACE_DOWNSCALE_FACTOR
        self.load_reference_face()

    def load_reference_face(self) -> bool:
        """Load and encode Rich's reference face from image file."""
        if face_recognition is None:
            print("⚠️ face_recognition library not installed.")
            return False

        if not self.reference_path.exists():
            print(f"ℹ️ Reference face not found at {self.reference_path}")
            self.reference_encoding = None
            return False

        try:
            image = face_recognition.load_image_file(str(self.reference_path))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                self.reference_encoding = encodings[0]
                print(f"✅ Loaded Rich's reference face from {self.reference_path}")
                return True
            else:
                print(f"⚠️ No face detected in reference image {self.reference_path}")
                self.reference_encoding = None
                return False
        except Exception as e:
            print(f"❌ Error loading reference face: {e}")
            self.reference_encoding = None
            return False

    def save_current_face(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
        """Save a cropped face from the current frame as rich.jpg and encode it."""
        try:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            # Add 20% margin around face crop
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            crop_x1 = max(0, x1 - margin_x)
            crop_y1 = max(0, y1 - margin_y)
            crop_x2 = min(w, x2 + margin_x)
            crop_y2 = min(h, y2 + margin_y)

            crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imwrite(str(self.reference_path), crop)
            print(f"📸 Saved new reference face to {self.reference_path}")
            return self.load_reference_face()
        except Exception as e:
            print(f"❌ Error saving reference face: {e}")
            return False

    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect and identify faces in the frame."""
        if face_recognition is None:
            return []

        # Downscale for faster detection
        small_frame = cv2.resize(
            frame,
            (0, 0),
            fx=self.downscale,
            fy=self.downscale,
            interpolation=cv2.INTER_LINEAR,
        )
        # Convert BGR (OpenCV) to RGB (face_recognition)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(
            rgb_small, model=config.FACE_DETECTION_MODEL
        )
        if not face_locations:
            return []

        # Compute encodings
        encodings = face_recognition.face_encodings(rgb_small, face_locations)

        detections: List[FaceDetection] = []
        inv_scale = 1.0 / self.downscale

        for (top, right, bottom, left), enc in zip(face_locations, encodings):
            # Scale coordinates back to original frame size
            x1 = int(left * inv_scale)
            y1 = int(top * inv_scale)
            x2 = int(right * inv_scale)
            y2 = int(bottom * inv_scale)
            box = (x1, y1, x2, y2)

            is_rich = False
            dist = 1.0
            name = "Unknown"

            if self.reference_encoding is not None:
                dist = float(face_recognition.face_distance([self.reference_encoding], enc)[0])
                if dist <= self.tolerance:
                    is_rich = True
                    name = "Rich"
            else:
                # If no reference face exists, consider the primary detected face as candidate Rich
                name = "Unknown (Press C)"

            detections.append(FaceDetection(box=box, is_rich=is_rich, distance=dist, name=name))

        return detections


class PlateOcrDetector:
    """Detects rectangular cards/plates using YOLOv8 and reads text via EasyOCR."""

    def __init__(self, yolo_path: str = config.YOLO_MODEL_PATH, device: str = config.DEVICE):
        self.device = device
        self.yolo_model = None
        self.ocr_reader = None
        self._init_models(yolo_path)

    def _init_models(self, yolo_path: str) -> None:
        """Initialize YOLO and EasyOCR models."""
        if YOLO is not None:
            try:
                print(f"🚀 Loading YOLOv8 on device: {self.device}")
                self.yolo_model = YOLO(yolo_path)
            except Exception as e:
                print(f"⚠️ Failed to load YOLO on {self.device}, falling back to CPU: {e}")
                self.device = "cpu"
                self.yolo_model = YOLO(yolo_path)

        if easyocr is not None:
            try:
                # EasyOCR handles PyTorch device internally; on Apple Silicon MPS or CPU
                use_gpu = torch.backends.mps.is_available()
                self.ocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
                print(f"✅ EasyOCR initialized (GPU/MPS enabled: {use_gpu})")
            except Exception as e:
                print(f"⚠️ EasyOCR GPU initialization fallback to CPU: {e}")
                self.ocr_reader = easyocr.Reader(["en"], gpu=False)

    def detect_pets(self, frame: np.ndarray) -> List[PetDetection]:
        """Detect cats and dogs in the frame using YOLOv8."""
        if self.yolo_model is None:
            return []
        pets: List[PetDetection] = []
        try:
            results = self.yolo_model.predict(
                frame,
                device=self.device,
                conf=config.YOLO_CONFIDENCE,
                verbose=False,
                classes=list(config.PET_CLASSES.keys()),
            )
            for res in results:
                for b in res.boxes:
                    cls_id = int(b.cls[0].item())
                    species = config.PET_CLASSES.get(cls_id, "pet")
                    coords = b.xyxy[0].cpu().numpy().astype(int)
                    bx1, by1, bx2, by2 = coords
                    conf = float(b.conf[0].item())
                    pets.append(PetDetection(box=(int(bx1), int(by1), int(bx2), int(by2)), species=species, confidence=conf))
        except Exception as e:
            print(f"⚠️ Pet detection error: {e}")
        return pets

    @staticmethod
    def _enhance_crop(crop: np.ndarray) -> np.ndarray:
        """Apply contrast-limited adaptive histogram equalization (CLAHE) to sharpen text."""
        if crop is None or crop.size == 0:
            return crop
        try:
            if len(crop.shape) == 3:
                lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
                l_chan, a_chan, b_chan = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l_chan)
                enhanced = cv2.merge((cl, a_chan, b_chan))
                return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(crop)
        except Exception:
            return crop

    def extract_candidate_regions(
        self, frame: np.ndarray, rich_box: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Tuple[int, int, int, int]]:
        """Extract candidate bounding boxes for cards/plates using YOLO + rectangular contours + lateral body regions."""
        h, w = frame.shape[:2]
        candidate_boxes: List[Tuple[int, int, int, int]] = []

        # 1. YOLO Detections (Books, cell phones, remotes, handheld items, cards, pets)
        if self.yolo_model is not None:
            try:
                results = self.yolo_model.predict(
                    frame,
                    device=self.device,
                    conf=config.YOLO_CONFIDENCE,
                    verbose=False,
                    classes=config.YOLO_TARGET_CLASSES,
                )
                for res in results:
                    for b in res.boxes:
                        coords = b.xyxy[0].cpu().numpy().astype(int)
                        bx1, by1, bx2, by2 = coords
                        candidate_boxes.append((int(bx1), int(by1), int(bx2), int(by2)))
            except Exception as e:
                print(f"⚠️ YOLO inference error: {e}")

        # 2. Rectangular Contour Analysis (cards, sheets, plates held up)
        # Search area: focus around Rich's upper body, torso, and hands
        roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, w, h
        if rich_box is not None:
            rx1, ry1, rx2, ry2 = rich_box
            rw = rx2 - rx1
            rh = ry2 - ry1
            # Expand region below and generously to sides of Rich's face (torso/hands area)
            roi_x1 = max(0, rx1 - int(rw * 2.2))
            roi_y1 = max(0, ry1 - int(rh * 0.5))
            roi_x2 = min(w, rx2 + int(rw * 2.2))
            roi_y2 = min(h, ry2 + int(rh * 3.5))

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size > 0:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Filter by area (card/plate size: ~1,500 to 200,000 px^2)
                if 1500 <= area <= 200000:
                    perimeter = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
                    # Rectangular polygon check (4-6 corners or near quadrilateral)
                    if len(approx) in [4, 5, 6]:
                        cx, cy, cw, ch = cv2.boundingRect(approx)
                        aspect_ratio = float(cw) / max(1, ch)
                        # License plates & cards: support both landscape and portrait orientations
                        if 0.35 <= aspect_ratio <= 5.0:
                            abs_box = (
                                roi_x1 + cx,
                                roi_y1 + cy,
                                roi_x1 + cx + cw,
                                roi_y1 + cy + ch,
                            )
                            candidate_boxes.append(abs_box)

        # 3. Guaranteed Handheld / Torso Regions (where plates/cards are held)
        if rich_box is not None:
            rx1, ry1, rx2, ry2 = rich_box
            rw = rx2 - rx1
            rh = ry2 - ry1
            # Center chest & torso area below chin
            chest_box = (
                max(0, rx1 - int(rw * 1.0)),
                min(h - 10, ry2),
                min(w, rx2 + int(rw * 1.0)),
                min(h, ry2 + int(rh * 2.5)),
            )
            candidate_boxes.append(chest_box)

            # Left hand lateral position
            left_hand_box = (
                max(0, rx1 - int(rw * 2.2)),
                max(0, ry1),
                rx1,
                min(h, ry2 + int(rh * 2.2)),
            )
            candidate_boxes.append(left_hand_box)

            # Right hand lateral position
            right_hand_box = (
                rx2,
                max(0, ry1),
                min(w, rx2 + int(rw * 2.2)),
                min(h, ry2 + int(rh * 2.2)),
            )
            candidate_boxes.append(right_hand_box)
        else:
            # Fallback: central region of frame where items are usually held up
            center_box = (
                int(w * 0.15),
                int(h * 0.20),
                int(w * 0.85),
                int(h * 0.85),
            )
            candidate_boxes.append(center_box)

        # 4. Non-Maximum Suppression / De-duplication on candidate boxes
        filtered_boxes = self._suppress_overlapping_boxes(candidate_boxes)
        return filtered_boxes[: config.MAX_OCR_CROPS]

    def _suppress_overlapping_boxes(
        self, boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.5
    ) -> List[Tuple[int, int, int, int]]:
        """Suppresses heavily overlapping boxes."""
        if not boxes:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = areas.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(boxes[i])

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def run_ocr_on_crops(
        self, frame: np.ndarray, candidate_boxes: List[Tuple[int, int, int, int]]
    ) -> List[CardDetection]:
        """Runs EasyOCR on candidate crops across both orientations (normal and flipped)
        with CLAHE contrast enhancement to handle macOS camera mirroring and prevent missed detections.
        """
        if self.ocr_reader is None or not candidate_boxes:
            return []

        h, w = frame.shape[:2]
        detections: List[CardDetection] = []

        for box in candidate_boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 30 or y2 - y1 < 20:
                continue

            crop = frame[y1:y2, x1:x2]
            enhanced = self._enhance_crop(crop)

            # Try normal orientation first
            crops_to_evaluate = [("normal", enhanced)]
            # Also evaluate horizontally flipped crop to resolve macOS reverse image
            crops_to_evaluate.append(("flipped", cv2.flip(enhanced, 1)))

            crop_detected = False
            for mode, current_crop in crops_to_evaluate:
                try:
                    ocr_results = self.ocr_reader.readtext(
                        current_crop,
                        detail=1,
                        text_threshold=config.OCR_TEXT_THRESHOLD,
                        low_text=config.OCR_LOW_TEXT,
                        link_threshold=config.OCR_LINK_THRESHOLD,
                    )
                    for bbox, text, conf in ocr_results:
                        is_trigger = config.matches_trigger(text)

                        detections.append(
                            CardDetection(
                                box=box,
                                text=text,
                                confidence=float(conf),
                                is_trigger=is_trigger,
                            )
                        )
                        if is_trigger:
                            crop_detected = True
                            break
                except Exception:
                    pass

                if crop_detected:
                    break

        return detections


class StateEngine:
    """Manages tracking smoothing, proximity check, and dynamic state transitions."""

    def __init__(self):
        self.smoothed_rich_box: Optional[List[float]] = None
        self.is_triggered: bool = False
        self.current_label: str = config.DEFAULT_LABEL
        self.last_trigger_time: float = 0.0
        self.last_rich_seen_time: float = 0.0
        self.active_card_box: Optional[Tuple[int, int, int, int]] = None
        self.smoothing_alpha = 0.45  # EMA smoothing factor

    def smooth_box(
        self, current_box: Optional[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Exponential Moving Average (EMA) to smooth bounding box motion across frames."""
        if current_box is None:
            self.smoothed_rich_box = None
            return None

        if self.smoothed_rich_box is None:
            self.smoothed_rich_box = [float(v) for v in current_box]
        else:
            for i in range(4):
                self.smoothed_rich_box[i] = (
                    self.smoothing_alpha * current_box[i]
                    + (1.0 - self.smoothing_alpha) * self.smoothed_rich_box[i]
                )

        return (
            int(self.smoothed_rich_box[0]),
            int(self.smoothed_rich_box[1]),
            int(self.smoothed_rich_box[2]),
            int(self.smoothed_rich_box[3]),
        )

    def is_adjacent_or_contained(
        self,
        person_box: Tuple[int, int, int, int],
        card_box: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
    ) -> bool:
        """Determines if the card is inside or directly adjacent to Rich's detected region."""
        px1, py1, px2, py2 = person_box
        cx1, cy1, cx2, cy2 = card_box
        _, frame_w = frame_shape

        # Centers
        p_center_x = (px1 + px2) / 2.0
        p_center_y = (py1 + py2) / 2.0
        c_center_x = (cx1 + cx2) / 2.0
        c_center_y = (cy1 + cy2) / 2.0

        # Normalized Euclidean distance between centers
        dist_x = abs(p_center_x - c_center_x)
        dist_y = abs(p_center_y - c_center_y)
        norm_dist = (dist_x**2 + dist_y**2) ** 0.5 / frame_w

        # Check containment or close proximity
        is_close = norm_dist <= config.MAX_PROXIMITY_RATIO
        return is_close

    def update(
        self,
        rich_box: Optional[Tuple[int, int, int, int]],
        card_detections: List[CardDetection],
        frame_shape: Tuple[int, int],
        now: float,
    ) -> Tuple[str, bool, Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
        """Update state given current detections and return:
        (label, is_triggered, smoothed_rich_box, active_card_box)
        """
        # Smooth Rich's bounding box
        smoothed_box = self.smooth_box(rich_box)

        trigger_found_in_frame = False
        triggered_card_box = None

        if smoothed_box is not None and card_detections:
            for card in card_detections:
                if card.is_trigger:
                    # Verify card is adjacent to or within Rich's zone
                    if self.is_adjacent_or_contained(smoothed_box, card.box, frame_shape):
                        trigger_found_in_frame = True
                        triggered_card_box = card.box
                        break

        # Dynamic State Transition
        if trigger_found_in_frame:
            self.is_triggered = True
            self.last_trigger_time = now
            self.current_label = config.TRIGGERED_LABEL
            self.active_card_box = triggered_card_box
        else:
            # Clean State Reset: Revert immediately when plate leaves frame or timeout expires
            time_since_trigger = now - self.last_trigger_time
            if time_since_trigger >= config.STATE_RESET_TIMEOUT:
                self.is_triggered = False
                self.current_label = config.DEFAULT_LABEL
                self.active_card_box = None

        return self.current_label, self.is_triggered, smoothed_box, self.active_card_box
