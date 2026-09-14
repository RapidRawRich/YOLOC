"""Integration & unit tests for the computer vision pipeline.
Tests model loading, MPS acceleration, EasyOCR parsing, regex matching,
and the StateEngine dynamic state transitions without requiring a physical camera.
"""

import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import torch

import config
from detector import (
    CardDetection,
    FaceDetection,
    FaceRecognitionManager,
    PlateOcrDetector,
    StateEngine,
)


def test_device_acceleration():
    print("\n--- Test 1: Device Acceleration (Apple Silicon MPS) ---")
    device = config.DEVICE
    print(f"Configured device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    print(f"MPS Built: {torch.backends.mps.is_built()}")

    # Test tensor allocation on MPS if available
    if torch.backends.mps.is_available():
        x = torch.ones((2, 2), device="mps")
        y = x * 2
        assert float(y[0, 0].cpu()) == 2.0, "MPS tensor multiplication failed"
        print("✅ MPS tensor operations verified successfully!")
    else:
        print("ℹ️ MPS not available, running on CPU.")


def test_state_engine_transitions():
    print("\n--- Test 2: StateEngine Dynamic State & Reset ---")
    engine = StateEngine()
    frame_shape = (720, 1280)
    rich_box = (400, 150, 600, 350)  # Face box in upper center
    now = time.time()

    # Step 1: Rich alone -> Default state
    label, triggered, smoothed_box, card_box = engine.update(
        rich_box=rich_box,
        card_detections=[],
        frame_shape=frame_shape,
        now=now,
    )
    assert label == "Rich", f"Expected 'Rich', got '{label}'"
    assert not triggered, "Expected triggered to be False"
    assert card_box is None, "Expected card_box to be None"
    print("✅ Step 1 passed: Rich alone produces default label 'Rich'")

    # Step 2: Rich + Card reading 'cunt' held adjacent/near torso -> Triggered state
    adjacent_card_box = (450, 380, 650, 480)  # Held right below face / near chest
    trigger_card = CardDetection(
        box=adjacent_card_box,
        text="CUNT",
        confidence=0.95,
        is_trigger=True,
    )

    label, triggered, smoothed_box, card_box = engine.update(
        rich_box=rich_box,
        card_detections=[trigger_card],
        frame_shape=frame_shape,
        now=now,
    )
    assert label == "Rich is a cunt", f"Expected 'Rich is a cunt', got '{label}'"
    assert triggered, "Expected triggered to be True"
    assert card_box == adjacent_card_box, "Expected card_box to match trigger card"
    print("✅ Step 2 passed: Target card near Rich triggers 'Rich is a cunt'")

    # Step 3: Card leaves frame -> Immediate reset
    now_later = now + config.STATE_RESET_TIMEOUT + 0.1
    label, triggered, smoothed_box, card_box = engine.update(
        rich_box=rich_box,
        card_detections=[],
        frame_shape=frame_shape,
        now=now_later,
    )
    assert label == "Rich", f"Expected reset to 'Rich', got '{label}'"
    assert not triggered, "Expected triggered to reset to False"
    assert card_box is None, "Expected card_box to reset to None"
    print("✅ Step 3 passed: Removing plate cleanly resets label back to 'Rich'")


def test_ocr_and_regex_matching():
    print("\n--- Test 3: OCR Trigger Word Regex Matching ---")
    test_cases = [
        ("cunt", True),
        ("CUNT", True),
        ("CVNT", True),     # OCR often mistakes U for V
        ("c*nt", True),
        ("C0NT", True),
        ("c u n t", True),  # With whitespace
        ("TMUJ", True),     # Mirrored font OCR artifact
        ("TNUC", True),     # Reversed word
        ("tnvc", True),     # Mirrored + V substitution
        ("license plate", False),
        ("hello world", False),
        ("count", False),
    ]

    for text, expected in test_cases:
        matched = config.matches_trigger(text)
        assert matched == expected, f"Failed regex check for '{text}': got {matched}, expected {expected}"
        print(f"  '{text}' -> {'TRIGGER' if matched else 'IGNORE'} (as expected)")

    print("✅ All regex matching cases verified!")


def test_synthetic_ocr_pipeline():
    print("\n--- Test 4: End-to-End Dual-Orientation OCR (Normal & Mirrored) ---")
    try:
        import easyocr
    except ImportError:
        print("⚠️ easyocr not installed, skipping synthetic OCR test.")
        return

    # Create a synthetic image containing a white plate with black text "CUNT"
    synth_img = np.full((120, 400, 3), 245, dtype=np.uint8)  # Off-white plate
    cv2.rectangle(synth_img, (4, 4), (396, 116), (0, 0, 0), 3)  # Border
    cv2.putText(
        synth_img,
        "CUNT",
        (60, 85),
        cv2.FONT_HERSHEY_DUPLEX,
        2.5,
        (10, 10, 10),
        5,
        cv2.LINE_AA,
    )

    # Test 1: Normal image
    # Test 2: Mirrored image (simulating macOS reverse image feed)
    test_orientations = [
        ("Normal Plate", synth_img),
        ("Mirrored (macOS reversed) Plate", cv2.flip(synth_img, 1)),
    ]

    reader = easyocr.Reader(["en"], gpu=torch.backends.mps.is_available())

    for label, img in test_orientations:
        found_trigger = False
        # Dual-orientation check: as implemented in detector.py
        for crop in [img, cv2.flip(img, 1)]:
            results = reader.readtext(
                crop,
                detail=1,
                text_threshold=config.OCR_TEXT_THRESHOLD,
                low_text=config.OCR_LOW_TEXT,
                link_threshold=config.OCR_LINK_THRESHOLD,
            )
            for bbox, text, conf in results:
                if config.matches_trigger(text):
                    found_trigger = True
                    print(f"  [{label}] Extracted: '{text}' (conf: {conf:.2f}) -> TRIGGER MATCH!")
                    break
            if found_trigger:
                break

        assert found_trigger, f"Failed to detect trigger word from {label}"
    print("✅ Dual-orientation OCR successfully detects both normal and reversed images!")


def main():
    print("============================================================")
    print(" Running YOLOC Computer Vision Pipeline Verification Tests   ")
    print("============================================================")

    test_device_acceleration()
    test_state_engine_transitions()
    test_ocr_and_regex_matching()
    test_synthetic_ocr_pipeline()

    print("\n============================================================")
    print(" 🎉 All Integration Tests Passed Successfully!             ")
    print("============================================================")


if __name__ == "__main__":
    main()
