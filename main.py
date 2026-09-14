"""Main real-time computer vision application.
Features:
- Apple Silicon (MPS) accelerated object detection with YOLOv8.
- Multi-threaded asynchronous Face Recognition and EasyOCR pipeline.
- Real-time 30 FPS display loop with exponential smoothing.
- Dynamic state merging ("Rich" -> "Rich is a cunt") and instant reset.
- Interactive face enrollment with 'C' key.
"""

import argparse
import os
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from detector import (
    CardDetection,
    FaceDetection,
    FaceRecognitionManager,
    PlateOcrDetector,
    StateEngine,
)
import ui_renderer


class RealtimeCVApp:
    def __init__(
        self,
        camera_index: int = config.CAMERA_INDEX,
        device: str = config.DEVICE,
        flip_horizontal: bool = config.FLIP_HORIZONTAL,
    ):
        self.camera_index = camera_index
        self.device = device
        self.flip_horizontal = flip_horizontal

        print("🔧 Initializing modules...")
        self.face_manager = FaceRecognitionManager()
        self.ocr_detector = PlateOcrDetector(device=self.device)
        self.state_engine = StateEngine()

        # Threading state & synchronization
        self.running = threading.Event()
        self.running.set()

        self.lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.new_frame_event = threading.Event()

        # Detection results cache (shared between workers and main thread)
        self.detected_faces: List[FaceDetection] = []
        self.detected_cards: List[CardDetection] = []
        self.current_rich_box: Optional[Tuple[int, int, int, int]] = None

        # Worker threads
        self.face_thread = threading.Thread(target=self._face_worker, daemon=True)
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)

        # Performance telemetry
        self.prev_time = time.time()
        self.fps = 0.0
        self.fps_filter = 0.0

    def _face_worker(self) -> None:
        """Background worker thread for face detection & recognition."""
        while self.running.is_set():
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            # Run face detection
            faces = self.face_manager.detect_faces(frame)

            with self.lock:
                self.detected_faces = faces
                # Identify Rich box for OCR targeting
                rich_box = None
                for f in faces:
                    if f.is_rich:
                        rich_box = f.box
                        break
                # If rich.jpg is not yet configured, use primary detected face as guide
                if rich_box is None and len(faces) == 1 and self.face_manager.reference_encoding is None:
                    rich_box = faces[0].box
                self.current_rich_box = rich_box

            # Sleep slightly to avoid pegging CPU when idle
            time.sleep(0.04)

    def _ocr_worker(self) -> None:
        """Background worker thread for YOLO detection + EasyOCR text extraction."""
        while self.running.is_set():
            frame = None
            rich_box = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    rich_box = self.current_rich_box

            if frame is None:
                time.sleep(0.01)
                continue

            # Extract candidate regions via YOLO & rectangular contour analysis
            candidate_boxes = self.ocr_detector.extract_candidate_regions(frame, rich_box)

            # Run OCR on candidate crops
            cards = self.ocr_detector.run_ocr_on_crops(frame, candidate_boxes)

            with self.lock:
                self.detected_cards = cards

            time.sleep(0.05)

    def run(self) -> None:
        """Starts the capture loop and renders the UI."""
        print(f"🎥 Connecting to webcam (device {self.camera_index})...")
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open webcam at index {self.camera_index}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        # Start worker threads
        self.face_thread.start()
        self.ocr_thread.start()

        window_name = "YOLOC - Real-Time Computer Vision"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        print("\n" + "=" * 60)
        print(" System Running!")
        print(f" Hardware Acceleration : {self.device.upper()}")
        print(f" Reference Face Path   : {config.REFERENCE_IMAGE_PATH}")
        print(" Controls:")
        print("   [M] Toggle horizontal flip (mirror / un-reversed normal view)")
        print("   [C] Capture current face as Rich (rich.jpg)")
        print("   [S] Save screenshot")
        print("   [Q] / [ESC] Quit application")
        print("=" * 60 + "\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Failed to grab frame from webcam. Retrying...")
                    time.sleep(0.05)
                    continue

                # Apply horizontal flip if enabled to correct macOS mirroring
                if self.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                # Calculate display loop FPS
                now = time.time()
                dt = now - self.prev_time
                self.prev_time = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    self.fps_filter = 0.9 * self.fps_filter + 0.1 * inst_fps if self.fps_filter > 0 else inst_fps

                # Share frame with worker threads
                with self.lock:
                    self.latest_frame = frame
                    faces = list(self.detected_faces)
                    cards = list(self.detected_cards)
                    rich_box = self.current_rich_box

                # Update State Engine (merging & smoothing)
                label, triggered, smoothed_rich_box, active_card_box = self.state_engine.update(
                    rich_box, cards, frame.shape[:2], now
                )

                # Render UI
                display_frame = frame.copy()

                # 1. Render Rich's Bounding Box and Badge
                if smoothed_rich_box is not None:
                    box_color = config.COLOR_TRIGGERED if triggered else config.COLOR_DEFAULT
                    ui_renderer.draw_hud_bracket_box(
                        display_frame, smoothed_rich_box, box_color, thickness=3, corner_len=24
                    )
                    rx1, ry1, rx2, ry2 = smoothed_rich_box
                    # Draw text badge above box
                    badge_y = max(ry1 - 8, 48)
                    ui_renderer.draw_pill_badge(
                        display_frame,
                        label,
                        (rx1, badge_y),
                        bg_color=box_color,
                        text_color=config.COLOR_TEXT,
                        font_scale=0.75,
                        thickness=2,
                    )

                # 2. Render Detected Card/Plate (if present)
                if active_card_box is not None:
                    ui_renderer.draw_rounded_rect(
                        display_frame,
                        (active_card_box[0], active_card_box[1]),
                        (active_card_box[2], active_card_box[3]),
                        config.COLOR_CARD,
                        thickness=2,
                        radius=8,
                    )
                    ui_renderer.draw_pill_badge(
                        display_frame,
                        f"Target [{config.TRIGGER_WORD}]",
                        (active_card_box[0], active_card_box[1] - 6),
                        bg_color=config.COLOR_CARD,
                        text_color=(0, 0, 0),
                        font_scale=0.55,
                        thickness=2,
                    )

                # 3. Top HUD Header
                rich_is_detected = smoothed_rich_box is not None
                card_is_detected = active_card_box is not None
                ui_renderer.draw_hud_header(
                    display_frame,
                    self.fps_filter,
                    self.device,
                    rich_is_detected,
                    triggered,
                    card_is_detected,
                    flip_horizontal=self.flip_horizontal,
                )

                # 4. Prompt banner if reference face is missing
                if self.face_manager.reference_encoding is None:
                    ui_renderer.draw_guide_banner(
                        display_frame,
                        "Reference 'rich.jpg' not found! Align your face and press [C] to enroll as Rich.",
                    )

                # Display frame
                cv2.imshow(window_name, display_frame)

                # Keyboard controls
                key = cv2.waitKey(1) & 0xFF
                if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
                    break
                elif key in [ord("m"), ord("M")]:  # 'm' to toggle flip
                    self.flip_horizontal = not self.flip_horizontal
                    state_desc = "ON (Normal / Un-reversed)" if self.flip_horizontal else "OFF (Raw Camera)"
                    print(f"🔄 Horizontal Flip: {state_desc}")
                elif key in [ord("c"), ord("C")]:  # 'c' to capture face
                    if smoothed_rich_box is not None:
                        success = self.face_manager.save_current_face(frame, smoothed_rich_box)
                        if success:
                            print("🎉 Rich's face successfully registered and loaded!")
                    else:
                        print("⚠️ No face currently detected in frame. Look into camera and retry.")
                elif key in [ord("s"), ord("S")]:  # 's' to snapshot
                    snap_name = f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(snap_name, display_frame)
                    print(f"📸 Saved snapshot to {snap_name}")

        finally:
            print("🛑 Shutting down CV pipeline...")
            self.running.clear()
            cap.release()
            cv2.destroyAllWindows()
            self.face_thread.join(timeout=1.0)
            self.ocr_thread.join(timeout=1.0)
            print("✅ Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="Real-time Computer Vision with Face & Card Recognition")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Webcam device index (default: 0)")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Torch device: mps or cpu")
    parser.add_argument(
        "--flip",
        dest="flip",
        action="store_true",
        default=config.FLIP_HORIZONTAL,
        help="Flip webcam horizontally to correct reverse image (default)",
    )
    parser.add_argument(
        "--no-flip",
        dest="flip",
        action="store_false",
        help="Do not flip webcam horizontally",
    )
    args = parser.parse_args()

    app = RealtimeCVApp(camera_index=args.camera, device=args.device, flip_horizontal=args.flip)
    app.run()


if __name__ == "__main__":
    main()
