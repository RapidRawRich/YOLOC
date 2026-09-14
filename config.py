"""Configuration settings for the real-time CV pipeline."""

from pathlib import Path
import re
import warnings
import torch

# Filter known non-critical third-party deprecation notices
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
REFERENCE_IMAGE_PATH = BASE_DIR / "rich.jpg"
YOLO_MODEL_PATH = "yolov8n.pt"

# Hardware Acceleration
def get_torch_device() -> str:
    """Select MPS if available on Apple Silicon, otherwise CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_torch_device()

# Camera & Display
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
FLIP_HORIZONTAL = True  # Flips webcam horizontally to correct macOS reverse/mirroring

# Face Recognition Settings
FACE_MATCH_TOLERANCE = 0.55  # Stricter than default 0.6 to minimize false positives
FACE_DOWNSCALE_FACTOR = 0.5   # Scale factor for fast detection
FACE_DETECTION_MODEL = "hog"  # Fast CPU-friendly model for face_recognition

# YOLO Object Detection Settings
YOLO_CONFIDENCE = 0.20        # Sensitive detection for held items
# Handheld items, books, screens, cups, and pets (15: cat, 16: dog)
YOLO_TARGET_CLASSES = [15, 16, 24, 26, 28, 39, 41, 63, 64, 65, 66, 67, 73, 74, 76, 77]
PET_CLASSES = {15: "cat", 16: "dog"}
MAX_OCR_CROPS = 8             # Process up to 8 candidate crops per cycle

# EasyOCR Detection Sensitivity
OCR_TEXT_THRESHOLD = 0.30     # Lower threshold to detect text on plates/cards under webcam lighting
OCR_LOW_TEXT = 0.20
OCR_LINK_THRESHOLD = 0.25

# OCR Trigger Settings
EXCLUDED_WORDS = {"count", "counter", "country", "account", "discount", "cant", "can't", "cent", "client", "carpet", "trumpet", "puppet"}
# Matches cunt, cvnt, c*nt, c0nt, clint, ciint, cnt, cuht
OCR_TRIGGER_REGEX = re.compile(r"\b[ck](?:u{1,2}|v{1,2}|0|\*|@|li|ii|)[nmh]{1,2}[t7i]\b", re.IGNORECASE)
TRIGGER_WORD = "cunt"
ALT_TRIGGER_WORDS = {"pet", "pets"}

# Known OCR permutations when text is mirrored horizontally (e.g. TMUJ, TNUC, etc.)
MIRRORED_PERMUTATIONS = {"tmuj", "tnuc", "tnvc", "tuvc", "7nuc", "tmvc", "tmuc", "tnvj", "tuvj", "t3p", "tep"}


def matches_trigger(raw_text: str) -> bool:
    """Robust trigger word matching handling OCR noise, mirrored letters, asterisk masking, spacing, and PET token."""
    if not raw_text:
        return False
    lower = raw_text.lower().strip()

    # Direct substring checks for primary trigger
    if TRIGGER_WORD in lower:
        return True

    words = re.findall(r"[a-zA-Z0-9\*@]+", lower)
    for w in words:
        if w in EXCLUDED_WORDS:
            continue
        if TRIGGER_WORD in w:
            return True
        if w in ALT_TRIGGER_WORDS:
            return True
        if w in MIRRORED_PERMUTATIONS:
            return True
        if OCR_TRIGGER_REGEX.fullmatch(w):
            return True
        # Check reversed word (in case OCR read mirrored text backward)
        w_rev = w[::-1]
        if w_rev in MIRRORED_PERMUTATIONS or TRIGGER_WORD in w_rev or w_rev in ALT_TRIGGER_WORDS or OCR_TRIGGER_REGEX.fullmatch(w_rev):
            return True

    no_spaces = re.sub(r"[\s\.\-_]", "", lower)
    if no_spaces in EXCLUDED_WORDS:
        return False
    if TRIGGER_WORD in no_spaces or no_spaces in MIRRORED_PERMUTATIONS or no_spaces in ALT_TRIGGER_WORDS:
        return True
    if OCR_TRIGGER_REGEX.search(no_spaces):
        return True
    # Check reversed no_spaces
    rev_no_spaces = no_spaces[::-1]
    if TRIGGER_WORD in rev_no_spaces or rev_no_spaces in MIRRORED_PERMUTATIONS or rev_no_spaces in ALT_TRIGGER_WORDS:
        return True
    if OCR_TRIGGER_REGEX.search(rev_no_spaces):
        return True

    return False

# Dynamic State Labels
DEFAULT_LABEL = "Rich"
TRIGGERED_LABEL = "Rich is a cunt"

# Proximity & Timing Thresholds
# Reset state if target word is not re-detected within this timeframe
STATE_RESET_TIMEOUT = 0.35  # seconds
# Max distance (in pixels normalized by frame width) between Rich and card to trigger (generous for arm's length)
MAX_PROXIMITY_RATIO = 0.65

# Visual Theme (BGR Colors)
COLOR_DEFAULT = (0, 230, 118)      # Neon Emerald Green
COLOR_TRIGGERED = (40, 40, 255)    # Vibrant Danger Red
COLOR_CARD = (255, 170, 0)         # Electric Cyan / Sky Blue
COLOR_PET = (255, 105, 180)        # Vibrant Orchid Pink / Hot Pink
COLOR_TEXT = (255, 255, 255)       # White
COLOR_BG_DARK = (20, 20, 25)       # Dark Slate
COLOR_HUD = (180, 180, 180)        # Silver Gray
