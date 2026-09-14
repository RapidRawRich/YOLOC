"""UI Renderer for the real-time computer vision pipeline.
Renders sleek, modern HUD overlays, rounded bounding boxes, and status badges.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

import config


def draw_rounded_rect(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    radius: int = 12,
) -> None:
    """Draw a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    w = x2 - x1
    h = y2 - y1
    r = min(radius, w // 2, h // 2)

    if r <= 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        return

    # Draw straight line segments
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Draw corner arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


def draw_hud_bracket_box(
    img: np.ndarray,
    box: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    corner_len: int = 20,
) -> None:
    """Draw a sleek HUD-style corner bracket bounding box."""
    x1, y1, x2, y2 = box
    # Subtle border outline
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    # Thick corner brackets
    c = max(corner_len, min((x2 - x1) // 5, (y2 - y1) // 5))
    t = max(thickness, 2)

    # Top-Left
    cv2.line(img, (x1, y1), (x1 + c, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y1), (x1, y1 + c), color, t, cv2.LINE_AA)

    # Top-Right
    cv2.line(img, (x2, y1), (x2 - c, y1), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y1), (x2, y1 + c), color, t, cv2.LINE_AA)

    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + c, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x1, y2), (x1, y2 - c), color, t, cv2.LINE_AA)

    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - c, y2), color, t, cv2.LINE_AA)
    cv2.line(img, (x2, y2), (x2, y2 - c), color, t, cv2.LINE_AA)


def draw_pill_badge(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = config.COLOR_TEXT,
    font_scale: float = 0.7,
    thickness: int = 2,
    padding: int = 8,
) -> None:
    """Draw a stylish solid pill badge containing text."""
    x, y = org
    font = cv2.FONT_HERSHEY_DUPLEX

    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    badge_w = text_w + padding * 2
    badge_h = text_h + padding * 2

    x1 = max(0, x)
    y1 = max(0, y - badge_h)
    x2 = min(img.shape[1], x1 + badge_w)
    y2 = min(img.shape[0], y1 + badge_h)

    # Draw shadow
    shadow_offset = 2
    cv2.rectangle(
        img,
        (x1 + shadow_offset, y1 + shadow_offset),
        (x2 + shadow_offset, y2 + shadow_offset),
        (10, 10, 10),
        -1,
        cv2.LINE_AA,
    )

    # Draw badge background
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)

    # Draw centered text
    text_x = x1 + padding
    text_y = y1 + padding + text_h
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def draw_hud_header(
    img: np.ndarray,
    fps: float,
    device: str,
    rich_detected: bool,
    triggered: bool,
    card_detected: bool,
    flip_horizontal: bool = True,
) -> None:
    """Draw top HUD status banner with device status, FPS, and detection telemetry."""
    h, w = img.shape[:2]

    # Semi-transparent dark header bar
    header_h = 42
    sub_img = img[0:header_h, 0:w]
    dark_rect = np.zeros(sub_img.shape, dtype=np.uint8)
    cv2.addWeighted(sub_img, 0.4, dark_rect, 0.6, 0, sub_img)

    # Border line under header
    cv2.line(img, (0, header_h), (w, header_h), (80, 80, 90), 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    y_text = 27

    # Left: Title, Device Info, and Mirror Status
    flip_status = "FLIP: ON" if flip_horizontal else "FLIP: OFF"
    title = f"YOLOC | {device.upper()} | {flip_status}"
    cv2.putText(img, title, (18, y_text), font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    # Center Status
    if triggered:
        status_text = "● TRIGGER ACTIVE: [Rich is a cunt]"
        status_color = config.COLOR_TRIGGERED
    elif rich_detected:
        status_text = "● TRACKING: Rich"
        status_color = config.COLOR_DEFAULT
    else:
        status_text = "○ SEARCHING..."
        status_color = (150, 150, 150)

    (st_w, _), _ = cv2.getTextSize(status_text, font, font_scale, 2)
    cv2.putText(img, status_text, ((w - st_w) // 2, y_text), font, font_scale, status_color, 2, cv2.LINE_AA)

    # Right: FPS meter and hotkey hints
    fps_text = f"FPS: {fps:04.1f} | [M] Flip"
    fps_color = (0, 255, 120) if fps >= 25 else (0, 180, 255)
    (fps_w, _), _ = cv2.getTextSize(fps_text, font, font_scale, 1)
    cv2.putText(img, fps_text, (w - fps_w - 20, y_text), font, font_scale, fps_color, 1, cv2.LINE_AA)


def draw_guide_banner(img: np.ndarray, message: str) -> None:
    """Draw a bottom banner instructing the user (e.g. how to capture face)."""
    h, w = img.shape[:2]
    banner_h = 36
    sub_img = img[h - banner_h : h, 0:w]
    dark_rect = np.full(sub_img.shape, 30, dtype=np.uint8)
    cv2.addWeighted(sub_img, 0.3, dark_rect, 0.7, 0, sub_img)

    cv2.line(img, (0, h - banner_h), (w, h - banner_h), (100, 100, 110), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), _ = cv2.getTextSize(message, font, 0.5, 1)
    cv2.putText(
        img,
        message,
        ((w - text_w) // 2, h - 12),
        font,
        0.5,
        (255, 230, 100),
        1,
        cv2.LINE_AA,
    )
