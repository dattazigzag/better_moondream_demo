"""
Image annotation renderer.

Takes detection/pointing results from Moondream and draws them onto
images — bounding boxes for detect, crosshair markers for point.
Returns annotated PIL Images ready for display in the Gradio chat.

All coordinates from Moondream are normalized (0–1 range). This module
converts them to pixel coordinates based on the actual image dimensions
before drawing.
"""

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
