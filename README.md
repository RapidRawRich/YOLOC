# YOLOC: Real-Time Computer Vision on macOS (Apple Silicon)

An asynchronous real-time computer vision system built with **Python**, **OpenCV**, **YOLOv8** (Apple Silicon `mps` accelerated), **EasyOCR**, and **Face Recognition**.

---

## Features

- **Decoupled 30 FPS Rendering Pipeline**: Deep learning inference (Face Recognition & OCR) runs asynchronously in dedicated background threads, ensuring the OpenCV webcam display loop runs smoothly at 30 FPS without stuttering.
- **Apple Silicon MPS Hardware Acceleration**: Uses PyTorch Metal Performance Shaders (`device='mps'`) for YOLOv8 object detection.
- **Target Face Identification**: Recognizes **Rich** from reference photo (`rich.jpg`).
- **Dynamic State Merging**:
  - **Default State**: Rich detected alone $\rightarrow$ Bounding box labeled **"Rich"** (sleek neon emerald badge).
  - **Triggered State**: When a card/plate reading **"cunt"** is detected near Rich $\rightarrow$ Bounding box label dynamically updates to **"Rich is a cunt"** (vibrant danger red badge).
  - **Instant Clean Reset**: Reverts immediately to **"Rich"** as soon as the card leaves the frame or the word is no longer detected.
- **Interactive Face Enrollment**:
  - If `rich.jpg` is missing, you can enroll your face instantly by pressing `C` in the live camera window.
  - Or use the standalone `capture_rich.py` utility.
- **Sleek HUD Overlay**: Anti-aliased corner brackets, pill-shaped glowing badges, real-time FPS counter, and device status telemetry.

---

## Quick Start

### 1. Automated Setup
Run the included setup script. It will verify macOS requirements, create a Python virtual environment, install dependencies, and pre-cache model weights:

```bash
chmod +x setup.sh
./setup.sh
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Enroll Rich's Face (Optional if you already have `rich.jpg`)
You can enroll Rich's face in either of two ways:
- **Option A (Interactive utility)**:
  ```bash
  python capture_rich.py
  ```
  Look into the webcam and press `C` or `Space` to save `rich.jpg`.
- **Option B (Existing photo)**:
  ```bash
  python capture_rich.py --image path/to/photo.jpg
  ```
- **Option C (On the fly in main app)**:
  Launch `main.py` and press `C` while looking into the camera.

### 4. Run the Real-Time Application
```bash
python main.py
```

Optional arguments:
```bash
python main.py --camera 0 --device mps
```

---

## Keyboard Controls

| Key | Action |
|---|---|
| **`M`** | Toggle horizontal flip (switch between mirror and un-reversed view) |
| **`C`** | Capture current face and save/reload as `rich.jpg` |
| **`S`** | Save screenshot to current directory |
| **`Q`** or **`ESC`** | Quit application |

---

## Verification & Tests

Run the headless integration test harness to verify Apple Silicon MPS acceleration, OCR token matching, and the state engine without requiring a physical camera:

```bash
python test_pipeline.py
```

---

## Project Structure

```
YOLOC/
├── requirements.txt      # Pinned dependencies
├── setup.sh              # Automated environment setup script
├── config.py             # Hyperparameters, regex patterns, colors, and camera settings
├── detector.py           # Face recognition, YOLO plate detection, OCR & StateEngine
├── ui_renderer.py        # Sleek HUD overlays, corner brackets, pill badges, and banners
├── capture_rich.py       # Standalone face enrollment utility
├── main.py               # Main multi-threaded 30 FPS capture & display application
├── test_pipeline.py      # Integration and headless test suite
└── README.md             # Documentation
```
