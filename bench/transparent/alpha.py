"""Model-free alpha helpers on float RGB arrays in [0, 1]: keys, the difference matte, the hybrid and trimaps."""

from __future__ import annotations

import numpy as np

_BORDER = 8


def border_colour(rgb: np.ndarray) -> np.ndarray:
    """Return the median colour of the outer *_BORDER* pixels, taken as the background colour."""
    edges = np.concatenate([rgb[:_BORDER].reshape(-1, 3), rgb[-_BORDER:].reshape(-1, 3), rgb[:, :_BORDER].reshape(-1, 3), rgb[:, -_BORDER:].reshape(-1, 3)])
    return np.median(edges, axis=0)


KEYS: dict[str, tuple[int, ...]] = {"green": (1,), "blue": (2,), "magenta": (0, 2), "cyan": (1, 2)}
"""Key colours by name, as the RGB channels that are high in the key ("on"); the rest are "off"."""


def key_excess(rgb: np.ndarray, on: tuple[int, ...]) -> np.ndarray:
    """Return how far each pixel leans to the key: its lowest on-channel minus its highest off-channel."""
    off = tuple(channel for channel in range(3) if channel not in on)
    return rgb[..., list(on)].min(axis=-1) - rgb[..., list(off)].max(axis=-1)


def key_channels(rgb: np.ndarray) -> tuple[int, ...]:
    """Return the on-channels of the key a render was made on, read from its border colour.

    Raises:
        ValueError: When the border is grey, white or black, so there is no key colour to read.
    """
    border = border_colour(rgb)
    on = tuple(channel for channel in range(3) if border[channel] > 0.5 * border.max())
    if len(on) == 3 or border.max() - border.min() < 0.2:
        raise ValueError(f"Border colour {np.round(border, 2).tolist()} is not a key colour.")
    return on


def chroma_key(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha, foreground)`` for a render on any key colour, by colour difference and despill.

    The key is read from the border, and the border's own excess sets the key strength, so a key that is not pure
    still works: pixels with under 15 % of that excess are opaque, pixels with over 70 % are clear. Despill removes the
    excess from the on-channels, which for green is ``G = min(G, max(R, B))``.
    """
    on = key_channels(rgb)
    excess = key_excess(rgb, on)
    key = max(float(np.median(_border_values(excess))), 1e-3)
    low, high = 0.15 * key, 0.7 * key
    alpha = 1.0 - np.clip((excess - low) / (high - low), 0.0, 1.0)
    foreground = rgb.copy()
    foreground[..., list(on)] -= np.clip(excess, 0.0, None)[..., None]
    return alpha, np.clip(foreground, 0.0, 1.0)


def key_conflict(rgb: np.ndarray, mask: np.ndarray, on: tuple[int, ...]) -> float:
    """Return the share of subject pixels (*mask* over half) that a key with *on*-channels would cut into."""
    subject = mask > 0.5
    return float((key_excess(rgb, on)[subject] > 0.12).mean()) if subject.any() else 0.0


def ink_key(rgb: np.ndarray, *, keep_enclosed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha, foreground)`` for dark ink on a light background, from each pixel's darkening of the paper.

    Alpha is the largest per-channel darkening relative to the border colour, so thin strokes keep whatever coverage
    they have and nothing is thresholded away. With *keep_enclosed*, background-coloured areas that don't touch the
    border (the inside of a letter or an outline) stay opaque, which suits logos and badges.
    """
    paper = np.maximum(border_colour(rgb), 1e-3)
    alpha = np.clip(((paper - rgb) / paper).max(axis=-1), 0.0, 1.0)
    safe = np.maximum(alpha, 1e-3)[..., None]
    foreground = np.clip(paper - (paper - rgb) / safe, 0.0, 1.0)
    if keep_enclosed:
        enclosed = _enclosed(alpha < 0.5)
        alpha = np.where(enclosed, 1.0, alpha)
        foreground = np.where(enclosed[..., None], rgb, foreground)
    return alpha, foreground


def screen_key(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(alpha, foreground)`` for light matter (smoke, glow, sparks) on a dark background.

    Alpha is the largest per-channel brightening over the border colour, so it assumes the subject's brightest
    colour is near white; coloured or light-absorbing subjects come out too transparent.
    """
    dark = border_colour(rgb)
    alpha = np.clip(((rgb - dark) / np.maximum(1.0 - dark, 1e-3)).max(axis=-1), 0.0, 1.0)
    safe = np.maximum(alpha, 1e-3)[..., None]
    return alpha, np.clip(dark + (rgb - dark) / safe, 0.0, 1.0)


def difference_matte(on_white: np.ndarray, on_black: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(alpha, foreground, residual)`` from one subject composited over a light and a dark background.

    With ``C = aF + (1 - a)B``, ``C_w - C_k = (1 - a)(B_w - B_k)`` per channel. The real backgrounds are read from the
    borders. *residual* is the per-pixel spread of the three channels' estimates of ``1 - a``: near zero when the
    subject is the same in both renders, large where the model redrew it.
    """
    light, dark = border_colour(on_white), border_colour(on_black)
    span = np.maximum(light - dark, 1e-3)
    clear = np.clip((on_white - on_black) / span, 0.0, 1.0)
    alpha = 1.0 - clear.mean(axis=-1)
    residual = clear.std(axis=-1)
    safe = np.maximum(alpha, 1e-3)[..., None]
    foreground = np.clip((on_black - (1.0 - alpha)[..., None] * dark) / safe, 0.0, 1.0)
    return alpha, foreground, residual


def hybrid_alpha(core: np.ndarray, pair_alpha: np.ndarray, residual: np.ndarray, *, tolerance: float) -> np.ndarray:
    """Return the pair's alpha where the pair agrees with itself, and the matting model's *core* alpha elsewhere.

    The difference matte is exact for see-through matter when both renders hold the same subject, and *residual*
    says where they don't: a model that relit or redrew the subject makes the three channels disagree. Pixels over
    *tolerance* fall back to *core*, so drifted hair keeps the model's edge while glass and smoke keep the pair's.
    """
    return np.where(residual <= tolerance, pair_alpha, core)


def subject_drift(on_white: np.ndarray, on_black: np.ndarray, core: np.ndarray) -> float:
    """Return the mean colour change between a pair's renders inside the solid part of the subject.

    Opaque pixels look the same over any background, so any change there is the model redrawing the subject.
    """
    import cv2

    solid = cv2.erode((core > 0.95).astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    return float(np.abs(on_white - on_black)[solid].mean()) if solid.any() else float("nan")


def trimap(mask: np.ndarray, *, radius: int) -> np.ndarray:
    """Return a trimap (0 background, 0.5 unknown, 1 foreground) with an unknown band of *radius* around the edge.

    Pixels the mask already leaves soft (between 0.05 and 0.95) stay unknown too, so hair the mask half-kept is
    left for the matting model.
    """
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    sure_fg = cv2.erode((mask > 0.95).astype(np.uint8), kernel).astype(bool)
    sure_bg = cv2.erode((mask < 0.05).astype(np.uint8), kernel).astype(bool)
    result = np.full(mask.shape, 0.5, dtype=np.float32)
    result[sure_fg] = 1.0
    result[sure_bg] = 0.0
    return result


def _enclosed(clear: np.ndarray) -> np.ndarray:
    import cv2

    count, labels = cv2.connectedComponents(clear.astype(np.uint8), connectivity=4)
    outside = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    return clear & ~np.isin(labels, outside)


def _border_values(channel: np.ndarray) -> np.ndarray:
    return np.concatenate([channel[:_BORDER].ravel(), channel[-_BORDER:].ravel(), channel[:, :_BORDER].ravel(), channel[:, -_BORDER:].ravel()])
