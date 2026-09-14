#!/usr/bin/env bash
# setup.sh - Automated setup for YOLOC real-time computer vision on macOS Apple Silicon

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " Setting up YOLOC: Real-Time Computer Vision on Apple Silicon "
echo "============================================================"

# 1. Check OS and Architecture
ARCH=$(uname -m)
OS=$(uname -s)

if [ "$OS" != "Darwin" ]; then
    echo "⚠️ Warning: This setup is optimized for macOS. Current OS: $OS"
fi

if [ "$ARCH" != "arm64" ]; then
    echo "⚠️ Warning: Recommended architecture is Apple Silicon (arm64). Current: $ARCH"
fi

# 2. Check for UV or Pip Package Manager
UV_BIN=""
if command -v uv &> /dev/null; then
    UV_BIN="$(command -v uv)"
elif [ -f "$HOME/.local/bin/uv" ]; then
    UV_BIN="$HOME/.local/bin/uv"
elif [ -f "$HOME/.cargo/bin/uv" ]; then
    UV_BIN="$HOME/.cargo/bin/uv"
fi

if [ -n "$UV_BIN" ]; then
    echo "⚡ Astral uv package manager detected: $UV_BIN"
fi

# Check Homebrew & CMake (needed for dlib compilation if building from source)
if ! command -v cmake &> /dev/null; then
    echo "📦 cmake not found. Installing via Homebrew..."
    if command -v brew &> /dev/null; then
        brew install cmake
    else
        echo "❌ Homebrew is not installed. Please install Homebrew or CMake manually."
        exit 1
    fi
else
    echo "✅ CMake found: $(command -v cmake)"
fi

# 3. Create Python Virtual Environment
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    if [ -n "$UV_BIN" ]; then
        echo "🐍 Creating virtual environment with uv at $VENV_DIR..."
        "$UV_BIN" venv "$VENV_DIR"
    else
        echo "🐍 Creating virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi
else
    echo "✅ Existing virtual environment found at $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# 4. Install & Upgrade Dependencies
if [ -n "$UV_BIN" ]; then
    echo "📦 Installing project dependencies with uv..."
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" -r requirements.txt
    # Ensure GUI OpenCV is installed
    "$UV_BIN" pip install --python "$VENV_DIR/bin/python" --reinstall "opencv-python>=4.8.0"
else
    echo "📦 Upgrading pip, setuptools (<82), and wheel..."
    pip install --upgrade pip wheel "setuptools<82"
    echo "📦 Installing project dependencies from requirements.txt..."
    pip install -r requirements.txt
    # Reinstall opencv-python to guarantee GUI Cocoa support isn't shadowed by headless build
    pip install --force-reinstall --no-deps "opencv-python>=4.8.0"
fi

# 6. Verify Apple Silicon (MPS) Acceleration
echo "🔍 Verifying PyTorch Apple Silicon (MPS) acceleration..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
mps_available = torch.backends.mps.is_available()
mps_built = torch.backends.mps.is_built()
print(f'MPS Available: {mps_available} | MPS Built: {mps_built}')
if mps_available:
    print('🚀 Apple Silicon MPS acceleration is ready!')
else:
    print('⚠️ MPS acceleration not available, fallback to CPU.')
"

# 7. Warm-up YOLO and EasyOCR models (downloads weights once now)
echo "📥 Pre-downloading YOLOv8 model..."
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
print('✅ YOLOv8n downloaded and cached.')
"

echo "📥 Pre-downloading EasyOCR English model..."
python -c "
import easyocr
reader = easyocr.Reader(['en'], gpu=False) # downloads weights
print('✅ EasyOCR weights downloaded and cached.')
"

echo ""
echo "============================================================"
echo " Setup Complete! 🎉"
echo "============================================================"
echo "To run the application:"
echo "  1. Activate virtual environment: source .venv/bin/activate"
echo "  2. (Optional) Enroll Rich's face: python capture_rich.py"
echo "  3. Start live stream:            python main.py"
echo "============================================================"
