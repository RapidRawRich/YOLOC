"""Face enrollment utility for Rich.
Run this script to capture and save a high-quality reference face (rich.jpg)
from your webcam or an existing image file.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

try:
    import face_recognition
except ImportError:
    face_recognition = None

import config


def capture_from_file(image_path: str, output_path: Path) -> bool:
    """Load an existing image file and verify/save as rich.jpg."""
    path = Path(image_path)
    if not path.exists():
        print(f"❌ File not found: {image_path}")
        return False

    img = cv2.imread(str(path))
    if img is None:
        print(f"❌ Could not read image from {image_path}")
        return False

    if face_recognition is not None:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        if not locations:
            print("⚠️ Warning: No faces detected in the provided image.")
        else:
            print(f"✅ Detected {len(locations)} face(s) in reference image.")

    cv2.imwrite(str(output_path), img)
    print(f"🎉 Successfully saved reference face to {output_path}")
    return True


def capture_from_webcam(
    camera_index: int, output_path: Path, flip_horizontal: bool = config.FLIP_HORIZONTAL
) -> bool:
    """Open webcam, guide user with bounding box, and save on 'C' or Spacebar."""
    print("🎥 Opening webcam...")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open webcam index {camera_index}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n------------------------------------------------------------")
    print(" Instructions:")
    print("   - Look directly into the camera.")
    print("   - Press [M] to toggle horizontal flip (un-reversed / mirror).")
    print("   - Press [C] or [SPACE] to capture your face.")
    print("   - Press [Q] or [ESC] to quit.")
    print("------------------------------------------------------------\n")

    window_name = "Enroll Face: Rich"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame from webcam.")
            break

        if flip_horizontal:
            frame = cv2.flip(frame, 1)

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Draw target guide ellipse / box in center
        center_x, center_y = w // 2, h // 2
        box_w, box_h = 240, 320
        x1 = center_x - box_w // 2
        y1 = center_y - box_h // 2
        x2 = center_x + box_w // 2
        y2 = center_y + box_h // 2

        # Draw guide overlay
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 230, 118), 2, cv2.LINE_AA)
        cv2.putText(
            display_frame,
            "Align face here and press 'C' to save",
            (center_x - 180, y1 - 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.6,
            (0, 230, 118),
            1,
            cv2.LINE_AA,
        )

        # Instructions banner at bottom
        flip_str = "ON (Normal)" if flip_horizontal else "OFF (Raw)"
        cv2.putText(
            display_frame,
            f"Keys: [C/Space] Capture | [M] Flip: {flip_str} | [Q] Quit",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord("c"), ord("C"), 32]:  # 'c', 'C', or Spacebar
            # Check face detection
            if face_recognition is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                locs = face_recognition.face_locations(rgb)
                if not locs:
                    print("⚠️ No face detected in frame. Please reposition and try again.")
                    continue
                print(f"✅ Found {len(locs)} face(s). Saving reference photo...")

            cv2.imwrite(str(output_path), frame)
            print(f"🎉 Saved reference face for Rich to: {output_path.resolve()}")
            saved = True
            break
        elif key in [ord("m"), ord("M")]:  # 'm' to toggle flip
            flip_horizontal = not flip_horizontal
            mode = "ON (Normal)" if flip_horizontal else "OFF (Raw Camera)"
            print(f"🔄 Horizontal Flip: {mode}")
        elif key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
            print("Operation cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved


def main():
    parser = argparse.ArgumentParser(description="Enroll reference face for Rich.")
    parser.add_argument("--image", type=str, help="Path to an existing photo of Rich")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index")
    parser.add_argument(
        "--flip",
        dest="flip",
        action="store_true",
        default=config.FLIP_HORIZONTAL,
        help="Flip webcam horizontally to correct reverse image",
    )
    parser.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        help="Do not flip webcam horizontally",
    )
    args = parser.parse_args()

    out_path = config.REFERENCE_IMAGE_PATH

    if args.image:
        success = capture_from_file(args.image, out_path)
    else:
        success = capture_from_webcam(args.camera, out_path, flip_horizontal=args.flip)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
