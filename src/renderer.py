"""
Image annotation renderer.

Takes detection/pointing/segmentation results from Moondream and draws
them onto images — bounding boxes for detect, crosshair markers for
point, and colored masks for segment.
Returns annotated PIL Images ready for display in the Gradio chat.

All coordinates from Moondream are normalized (0–1 range) except
segment paths which are in pixel coordinates. This module converts
normalized coordinates to pixel coordinates based on the actual image
dimensions before drawing.
"""

import re

from PIL import Image, ImageDraw, ImageFont


# Distinct colors for multiple detected objects. Each is (R, G, B).
BBOX_COLORS = [
    (255, 75, 75),  # red
    (75, 150, 255),  # blue
    (75, 220, 75),  # green
    (255, 200, 50),  # yellow
    (200, 100, 255),  # purple
    (255, 150, 50),  # orange
    (50, 220, 220),  # cyan
    (255, 100, 175),  # pink
]

POINT_COLOR = (255, 75, 75)  # red markers
POINT_RING_COLOR = (255, 255, 255)  # white outer ring for contrast


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """
    Try to load a clean font, fall back to PIL's built-in default.

    We don't bundle fonts — this just tries common system paths
    and gracefully degrades.
    """
    paths_to_try = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
    ]
    for path in paths_to_try:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_bounding_boxes(
    image: Image.Image,
    objects: list[dict],
    subject: str = "object",
) -> Image.Image:
    """
    Draw bounding boxes on an image for detected objects.

    Each object dict should have keys: x_min, y_min, x_max, y_max
    with values normalized to 0–1.

    Args:
        image: Original image (not modified).
        objects: List of bounding box dicts from MoondreamClient.detect().
        subject: Label text to show on each box.

    Returns:
        A new annotated PIL Image with boxes drawn on it.
    """
    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _get_font(size=max(14, image.width // 40))

    w, h = image.size

    for i, obj in enumerate(objects):
        color = BBOX_COLORS[i % len(BBOX_COLORS)]

        # Convert normalized coords to pixels
        x_min = int(obj["x_min"] * w)
        y_min = int(obj["y_min"] * h)
        x_max = int(obj["x_max"] * w)
        y_max = int(obj["y_max"] * h)

        # Semi-transparent fill
        fill_color = color + (40,)  # low alpha
        draw.rectangle([x_min, y_min, x_max, y_max], fill=fill_color)

        # Solid border (3px)
        outline_color = color + (220,)
        draw.rectangle([x_min, y_min, x_max, y_max], outline=outline_color, width=3)

        # Label background + text
        label = f"{subject} #{i + 1}" if len(objects) > 1 else subject
        bbox = font.getbbox(label)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        padding = 4

        label_bg = [
            x_min,
            y_min - text_h - padding * 2,
            x_min + text_w + padding * 2,
            y_min,
        ]
        # If label would go above image, put it inside the box
        if label_bg[1] < 0:
            label_bg[1] = y_min
            label_bg[3] = y_min + text_h + padding * 2

        draw.rectangle(label_bg, fill=color + (200,))
        draw.text(
            (label_bg[0] + padding, label_bg[1] + padding),
            label,
            fill=(255, 255, 255, 255),
            font=font,
        )

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")


def draw_points(
    image: Image.Image,
    points: list[dict],
    subject: str = "object",
) -> Image.Image:
    """
    Draw point markers on an image for located objects.

    Each point dict should have keys: x, y with values normalized
    to 0–1. Draws a crosshair with a contrasting ring so it's
    visible on any background.

    Args:
        image: Original image (not modified).
        points: List of point dicts from MoondreamClient.point().
        subject: Label text to show next to each marker.

    Returns:
        A new annotated PIL Image with point markers drawn on it.
    """
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = _get_font(size=max(14, image.width // 40))

    w, h = image.size
    # Marker radius scales with image size
    radius = max(8, min(w, h) // 60)

    for i, pt in enumerate(points):
        cx = int(pt["x"] * w)
        cy = int(pt["y"] * h)

        # Outer white ring for contrast
        draw.ellipse(
            [cx - radius - 2, cy - radius - 2, cx + radius + 2, cy + radius + 2],
            outline=POINT_RING_COLOR,
            width=3,
        )
        # Inner colored ring
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=POINT_COLOR,
            width=3,
        )
        # Crosshair lines
        line_len = radius + 4
        draw.line(
            [(cx - line_len, cy), (cx + line_len, cy)],
            fill=POINT_COLOR,
            width=2,
        )
        draw.line(
            [(cx, cy - line_len), (cx, cy + line_len)],
            fill=POINT_COLOR,
            width=2,
        )

        # Label
        label = f"{subject} #{i + 1}" if len(points) > 1 else subject
        text_x = cx + radius + 6
        text_y = cy - radius

        # Draw text with a dark outline for readability
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((text_x + dx, text_y + dy), label, fill=(0, 0, 0), font=font)
        draw.text((text_x, text_y), label, fill=POINT_COLOR, font=font)

    return annotated


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FUTURE: Segment mask rendering — currently unused because
# Moondream Station doesn't support segment locally (cloud-only).
# Station returns: {'error': "Function 'segment' not available"}
# Kept for when Station adds local segment support.
# See also: client.py → MoondreamClient.segment()
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SEGMENT_COLOR = (75, 150, 255)  # blue mask overlay


def _parse_svg_path(path: str) -> list[tuple[float, float]]:
    """
    Extract (x, y) coordinate pairs from a simple SVG path string.

    Moondream's segment endpoint returns SVG path data (M/L/Z commands)
    in pixel coordinates. This parser handles the common subset:
    M (moveto), L (lineto), and Z (closepath). It also handles
    implicit lineto when coordinates follow M without an explicit L.

    Not a full SVG parser — just enough for the polygon masks that
    Moondream returns.
    """
    points: list[tuple[float, float]] = []

    # Split the path into tokens (commands and numbers)
    tokens = re.findall(r"[MLZmlz]|[-+]?\d*\.?\d+", path)

    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in ("M", "L", "m", "l"):
            # Skip the command letter, read coordinate pairs
            i += 1
            while i < len(tokens) and tokens[i] not in "MLZmlz":
                x = float(tokens[i])
                y = float(tokens[i + 1]) if i + 1 < len(tokens) else 0.0
                points.append((x, y))
                i += 2
        elif token in ("Z", "z"):
            i += 1
        else:
            # Bare number — treat as implicit lineto coordinate
            try:
                x = float(tokens[i])
                y = float(tokens[i + 1]) if i + 1 < len(tokens) else 0.0
                points.append((x, y))
                i += 2
            except (ValueError, IndexError):
                i += 1

    return points


def draw_segment_mask(
    image: Image.Image,
    path: str,
    label: str = "segment",
) -> Image.Image:
    """
    Draw a segmentation mask overlay on an image.

    The SVG path from Moondream's segment endpoint describes a polygon
    in pixel coordinates. This function fills that polygon with a
    semi-transparent colour and draws its outline.

    Args:
        image: Original image (not modified).
        path: SVG path string from MoondreamClient.segment().
        label: Optional label to display at the top of the mask.

    Returns:
        A new annotated PIL Image with the segment mask drawn on it.
    """
    points = _parse_svg_path(path)

    if len(points) < 3:
        # Not enough points to form a polygon — return image as-is
        return image.copy()

    annotated = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Draw the filled polygon mask
    fill_color = SEGMENT_COLOR + (60,)  # semi-transparent
    outline_color = SEGMENT_COLOR + (200,)
    draw.polygon(points, fill=fill_color, outline=outline_color)

    # Draw the outline with extra width for visibility
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        draw.line([p1, p2], fill=outline_color, width=2)

    # Add label near the top of the bounding area
    if label:
        font = _get_font(size=max(14, image.width // 40))
        # Find the topmost point for label placement
        min_y_point = min(points, key=lambda p: p[1])
        text_x = int(min_y_point[0])
        text_y = int(min_y_point[1]) - 20

        # Keep label within image bounds
        text_y = max(4, text_y)
        text_x = max(4, min(text_x, image.width - 80))

        # Dark outline for readability
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text(
                (text_x + dx, text_y + dy),
                label,
                fill=(0, 0, 0, 255),
                font=font,
            )
        draw.text(
            (text_x, text_y),
            label,
            fill=SEGMENT_COLOR + (255,),
            font=font,
        )

    annotated = Image.alpha_composite(annotated, overlay)
    return annotated.convert("RGB")
