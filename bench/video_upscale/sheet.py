"""Blind review sheet for the video-upscale bench: crops of the same frames from every method under shuffled letters."""

from __future__ import annotations

import html
import json
import random
import subprocess
from pathlib import Path

import numpy as np

# Where frames are taken, as fractions of the clip; extension joins are added by the caller.
FRAME_FRACTIONS = (0.1, 0.4, 0.7, 0.95)
CROP_FRACTION = 0.45


def frame_indices(frame_count: int, joins: list[int]) -> list[int]:
    """Return the sorted frame indices to compare: fixed fractions of the clip plus the frame after each join."""
    picks = {min(frame_count - 1, int(frame_count * fraction)) for fraction in FRAME_FRACTIONS}
    picks.update(join for join in joins if 0 < join < frame_count)
    return sorted(picks)


def blind_letters(methods: list[str], seed: str) -> dict[str, str]:
    """Return a letter per method, shuffled by *seed* so the same clip always gets the same key."""
    order = list(methods)
    random.Random(seed).shuffle(order)
    return {method: chr(ord("A") + index) for index, method in enumerate(order)}


def _read_frame(video: Path, index: int) -> np.ndarray:
    """Return frame *index* of *video* as uint8 RGB ``(H, W, 3)``."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(video)],
        check=True,
        capture_output=True,
        text=True,
    )
    width, height = (int(value) for value in probe.stdout.strip().split(","))
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", rf"select=eq(n\,{index})", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def _centre_crop(frame: np.ndarray) -> np.ndarray:
    """Return the centre CROP_FRACTION of *frame* on each side."""
    height, width = frame.shape[:2]
    crop_h, crop_w = int(height * CROP_FRACTION), int(width * CROP_FRACTION)
    top, left = (height - crop_h) // 2, (width - crop_w) // 2
    return frame[top : top + crop_h, left : left + crop_w]


def _pixel_double(frame: np.ndarray) -> np.ndarray:
    """Return *frame* enlarged 2x by repeating each pixel, so no smoothing is added."""
    return frame.repeat(2, axis=0).repeat(2, axis=1)


def laplacian_variance(frame: np.ndarray) -> float:
    """Return the variance of the 4-neighbour Laplacian of *frame*'s luma, a rough sharpness measure."""
    luma = frame.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    lap = -4 * luma[1:-1, 1:-1] + luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:]
    return float(lap.var())


def _write_png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(frame).save(path)


def build_sheet(run_dir: Path, clips: dict[str, dict], methods_by_clip: dict[str, list[str]], frame_counts: dict[str, int], sources_dir: Path) -> Path:
    """Write ``sheet.html`` (blind) and ``key.json`` (letters and sharpness) in *run_dir* and return the sheet's path.

    Each row starts with the source frame, enlarged 2x by pixel repetition and labelled, so added detail can be judged
    against what the source holds; the methods follow under their blind letters.

    Args:
        run_dir: The run's folder, holding ``<clip>/<method>.mp4``.
        clips: The clip cases, for the extension joins.
        methods_by_clip: The methods to show per clip.
        frame_counts: The source frame count per clip.
        sources_dir: The folder holding ``<clip>.mp4``, the source clips.
    """
    crops_dir = run_dir / "crops"
    crops_dir.mkdir(exist_ok=True)
    key: dict[str, dict] = {}
    sections = []
    for clip, methods in methods_by_clip.items():
        letters = blind_letters(methods, clip)
        joins = _joins(clips[clip])
        indices = frame_indices(frame_counts[clip], joins)
        sharpness: dict[str, float] = {}
        rows = []
        for index in indices:
            source = _centre_crop(_pixel_double(_read_frame(sources_dir / f"{clip}.mp4", index)))
            source_path = crops_dir / f"{clip}_{index}_source.png"
            _write_png(source, source_path)
            cells = [f'<figure class=source><img src="crops/{source_path.name}" loading="lazy"><figcaption>Source</figcaption></figure>']
            for method in sorted(methods, key=letters.__getitem__):
                frame = _read_frame(run_dir / clip / f"{method}.mp4", index)
                sharpness[method] = sharpness.get(method, 0.0) + laplacian_variance(frame) / len(indices)
                crop_path = crops_dir / f"{clip}_{index}_{letters[method]}.png"
                _write_png(_centre_crop(frame), crop_path)
                cells.append(f'<figure><img src="crops/{crop_path.name}" loading="lazy"><figcaption>{letters[method]}</figcaption></figure>')
            label = f"frame {index}" + (" (first frame after an extension join)" if index in joins else "")
            rows.append(f"<h3>{html.escape(label)}</h3><div class=row>{''.join(cells)}</div>")
        videos = "".join(
            f'<figure><video src="{clip}/{method}.mp4" muted loop playsinline></video><figcaption>{letters[method]}</figcaption></figure>' for method in sorted(methods, key=letters.__getitem__)
        )
        sections.append(f'<section><h2>{html.escape(clip)}</h2><button onclick="play(this)">Play all</button><div class=row>{videos}</div>{"".join(rows)}</section>')
        key[clip] = {"letters": letters, "sharpness": {method: round(value, 1) for method, value in sharpness.items()}}
    (run_dir / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    sheet = run_dir / "sheet.html"
    sheet.write_text(_PAGE.format(body="".join(sections)), encoding="utf-8")
    return sheet


def _joins(case: dict) -> list[int]:
    """Return the first frame index of each extension of *case* in the final clip."""
    joins, total = [], case["frames"]
    for extend in case.get("extends", []):
        joins.append(total)
        total += extend["frames"]
    return joins


_PAGE = """<!doctype html>
<meta charset=utf-8>
<title>Video upscale review</title>
<style>
body {{ font: 14px system-ui; margin: 24px; background: #111; color: #ddd; }}
.row {{ display: flex; gap: 8px; flex-wrap: wrap; }}
figure {{ margin: 0; flex: 1 1 0; min-width: 300px; }}
img, video {{ width: 100%; display: block; image-rendering: pixelated; }}
.source figcaption {{ color: #8ab4f8; }}
figcaption {{ text-align: center; font-weight: 600; padding: 4px; }}
section {{ margin-bottom: 48px; }}
</style>
<p>Pick the best letter per clip. Crops are the centre of each frame at full output size. Source is the original frame with each pixel doubled, not one of the choices. The key is in key.json.</p>
{body}
<script>
function play(button) {{
  const videos = button.parentElement.querySelectorAll("video");
  videos.forEach(v => {{ v.currentTime = 0; v.play(); }});
}}
</script>
"""
