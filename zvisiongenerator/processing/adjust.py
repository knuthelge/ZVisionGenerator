"""Apply the fifteen light and colour settings of the viewer's Adjust editor to an 8-bit RGB array."""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass, fields

import numpy as np

KR = 0.2126
KG = 0.7152
KB = 0.0722
EV_MAX = 2.0
CONTRAST_MAX = 1.5
WARMTH_STOPS = 0.5
TONE_TOWARD = 1.5
TONE_AWAY = 1.0
VIBRANCE_BOOST = 2.0
VIBRANCE_MUTE = 1.0
VIBRANCE_FALLOFF = 2.0
MIX_NAMES = ("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta")
# Oklab hue of each band's sRGB colour, rounded; orange is 55, nudged from #FF8000's 52.98 so skin tones sit on its centre.
MIX_CENTRES = (29, 55, 110, 142, 195, 264, 294, 328)
# Each side of a centre keeps full weight for this fraction of the gap to its neighbour.
MIX_FLAT = 0.25
# Oklab chroma over which boosting fades in; +-1 LSB of noise on an 8-bit grey stays below the start of the full boost.
MIX_FADE_START = 0.002
MIX_FADE_END = 0.006
# Boost gain at low chroma and at and above MIX_VIVID_CHROMA: +100 multiplies the saturation by 1 + gain.
MIX_BOOST_MUTED = 3.0
MIX_BOOST_VIVID = 1.0
MIX_VIVID_CHROMA = 0.2
MIX_MUTE = 1.0
MIX_MIN_CHROMA = 1e-6
_MIX_NO_LIMIT = 1e6

SETTING_MIN = -100
SETTING_MAX = 100

_CHUNK_PIXELS = 1 << 20
_ENCODE_INTERVALS = 4096
_LUMA = np.array([KR, KG, KB], dtype=np.float32)
_OKLAB_LMS = np.array(
    [[0.4122214708, 0.5363325363, 0.0514459929], [0.2119034982, 0.6806995451, 0.1073969566], [0.0883024619, 0.2817188376, 0.6299787005]],
    dtype=np.float32,
)
_OKLAB_A = np.array([1.9779984951, -2.4285922050, 0.4505937099], dtype=np.float32)
_OKLAB_B = np.array([0.0259040371, 0.7827717662, -0.8086757660], dtype=np.float32)
_BAND_EDGES = np.array([*MIX_CENTRES, MIX_CENTRES[0] + 360], dtype=np.float32)


@dataclass(frozen=True)
class LightSettings:
    """Fifteen integer settings, each in [-100, 100]; 0 leaves that quality unchanged.

    The eight `mix_*` settings each change the saturation of one colour family.
    """

    exposure: int = 0
    contrast: int = 0
    highlights: int = 0
    shadows: int = 0
    saturation: int = 0
    vibrance: int = 0
    warmth: int = 0
    mix_red: int = 0
    mix_orange: int = 0
    mix_yellow: int = 0
    mix_green: int = 0
    mix_aqua: int = 0
    mix_blue: int = 0
    mix_purple: int = 0
    mix_magenta: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not SETTING_MIN <= value <= SETTING_MAX:
                raise ValueError(f"{field.name} must be between {SETTING_MIN} and {SETTING_MAX}, got {value}")

    @property
    def is_neutral(self) -> bool:
        """Whether every setting is 0, so the pixels pass through unchanged."""
        return not any(astuple(self))

    @property
    def mix(self) -> tuple[int, ...]:
        """The eight colour-mixer amounts, in `MIX_NAMES` order."""
        return tuple(getattr(self, f"mix_{name}") for name in MIX_NAMES)


def adjust_light(rgb: np.ndarray, light: LightSettings) -> np.ndarray:
    """Return a new uint8 H x W x 3 array with the light settings applied.

    Alpha is never read; callers pass the colour channels only.

    Args:
        rgb: An 8-bit sRGB array of shape (height, width, 3).
        light: The fifteen settings.

    Returns:
        A new array of the same shape. A copy of the input when every setting is 0.

    Raises:
        ValueError: If `rgb` is not a uint8 array of shape (height, width, 3).
    """
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected a uint8 array of shape (height, width, 3), got {rgb.dtype} {rgb.shape}")
    if light.is_neutral:
        return rgb.copy()
    out = np.empty(rgb.shape, dtype=np.uint8)
    source = rgb.reshape(-1, 3)
    target = out.reshape(-1, 3)
    tables = _Tables.build(light)
    for start in range(0, source.shape[0], _CHUNK_PIXELS):
        stop = start + _CHUNK_PIXELS
        target[start:stop] = _adjust_chunk(source[start:stop], light, tables)
    return out


@dataclass(frozen=True)
class _Tables:
    """Lookup tables and gains shared by every chunk of one call."""

    decode: np.ndarray
    gains: np.ndarray
    encode: np.ndarray
    channel: np.ndarray
    mix: np.ndarray | None

    @classmethod
    def build(cls, light: LightSettings) -> _Tables:
        return cls(
            decode=_decode_table(),
            gains=_gains(light),
            encode=_encode_table(light.contrast),
            channel=_channel_table(light.contrast),
            mix=np.array(light.mix, dtype=np.float32) / np.float32(100.0) if any(light.mix) else None,
        )


def _adjust_chunk(chunk: np.ndarray, light: LightSettings, tables: _Tables) -> np.ndarray:
    if light.exposure or light.warmth or light.saturation or light.vibrance or tables.mix is not None:
        linear = tables.decode[chunk]
        if light.exposure or light.warmth:
            linear = np.clip(linear * tables.gains, 0.0, 1.0)
        if light.saturation or light.vibrance or tables.mix is not None:
            linear = _saturate(linear, 1.0 + light.saturation / 100.0, light.vibrance / 100.0, tables.mix)
        encoded = _lerp(tables.encode, linear)
    else:
        encoded = tables.channel[chunk]
    if light.highlights or light.shadows:
        encoded = _tone(encoded, light.highlights / 100.0, light.shadows / 100.0)
    return np.floor(encoded * 255.0 + 0.5).astype(np.uint8)


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * np.maximum(v, 0.0031308) ** (1.0 / 2.4) - 0.055)


def _decode_table() -> np.ndarray:
    return _srgb_to_linear(np.arange(256, dtype=np.float64) / 255.0).astype(np.float32)


def _gains(light: LightSettings) -> np.ndarray:
    k = light.warmth / 100.0
    warmth = np.array([2.0 ** (WARMTH_STOPS * k), 1.0, 2.0 ** (-WARMTH_STOPS * k)])
    luminance_gain = float(np.dot([KR, KG, KB], warmth))
    return (2.0 ** (EV_MAX * light.exposure / 100.0) * warmth / luminance_gain).astype(np.float32)


def _saturate(linear: np.ndarray, saturation: float, vibrance: float, mix: np.ndarray | None) -> np.ndarray:
    """Mix each pixel toward or away from its luminance by `saturation`, `vibrance` and the colour mixer.

    The chroma `S` that weights the vibrance, and the hue and chroma that weight the mixer, are read from the
    pixel before any of the three apply. `mix` holds the eight band amounts divided by 100, or None when all are 0.
    """
    luma = linear @ _LUMA
    if not vibrance and mix is None:
        return np.clip(luma[:, None] + saturation * (linear - luma[:, None]), 0.0, 1.0)
    factor = np.float32(saturation)
    if vibrance:
        factor = factor * _vibrance_factor(linear, vibrance)
    if mix is not None:
        factor = _mix_factor(linear, luma, factor, mix)
    return np.clip(luma[:, None] + factor[:, None] * (linear - luma[:, None]), 0.0, 1.0)


def _vibrance_factor(linear: np.ndarray, vibrance: float) -> np.ndarray:
    red, green, blue = linear[:, 0], linear[:, 1], linear[:, 2]
    peak = np.maximum(np.maximum(red, green), blue)
    low = np.minimum(np.minimum(red, green), blue)
    chroma = (peak - low) / np.where(peak > 0.0, peak, np.float32(1.0))
    kv = VIBRANCE_BOOST if vibrance > 0 else VIBRANCE_MUTE
    return (1.0 + kv * vibrance * (1.0 - chroma) ** VIBRANCE_FALLOFF).astype(np.float32)


def _mix_factor(linear: np.ndarray, luma: np.ndarray, base: np.ndarray | np.float32, mix: np.ndarray) -> np.ndarray:
    """Return each pixel's saturation factor once the colour mixer has changed `base`, the saturation and vibrance factor.

    A band amount above 0 boosts: the extra factor grows with the amount, is strongest on muted colours and fades in with
    Oklab chroma so that near-greys are never tinted. A soft minimum keeps the boost inside the room left before a channel
    clips, so the mixer never clips a colour or shifts its hue. An amount below 0 scales `base` down and does not fade.
    Where the hue is undefined (chroma at or below `MIX_MIN_CHROMA`) the factor is `base`.
    """
    q = np.cbrt(_mix_columns(linear, _OKLAB_LMS))
    a = q @ _OKLAB_A
    b = q @ _OKLAB_B
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % np.float32(360.0)
    amount = _band_amount(hue, mix)
    base = np.broadcast_to(np.asarray(base, dtype=np.float32), luma.shape)
    fade = _smoothstep((chroma - np.float32(MIX_FADE_START)) / np.float32(MIX_FADE_END - MIX_FADE_START))
    excess = np.maximum(amount, 0.0) * _boost_gain(chroma) * fade * base
    headroom = np.minimum(np.maximum(_gamut_factor(linear, luma) - base, 0.0), np.float32(_MIX_NO_LIMIT))
    norm = np.maximum(_fourth_root(_fourth_power(excess) + _fourth_power(headroom)), np.float32(1e-12))
    boost = np.where(excess > 0.0, excess * headroom / norm, 0.0)
    mute = np.where(amount < 0.0, 1.0 + np.float32(MIX_MUTE) * amount, 1.0)
    return np.where(chroma > MIX_MIN_CHROMA, (base + boost) * mute, base).astype(np.float32)


def _mix_columns(x: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return `x @ matrix.T` for an N x 3 array as three column multiply-adds, which is faster than a skinny matmul."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return r * matrix[:, 0] + g * matrix[:, 1] + b * matrix[:, 2]


def _fourth_power(x: np.ndarray) -> np.ndarray:
    """Return `x ** 4`, the `p` of the soft minimum that keeps the mixer's boost inside the sRGB gamut."""
    x2 = x * x
    return x2 * x2


def _fourth_root(x: np.ndarray) -> np.ndarray:
    """Return `x ** 0.25`, the inverse of `_fourth_power`."""
    return np.sqrt(np.sqrt(x))


def _smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _band_amount(hue: np.ndarray, mix: np.ndarray) -> np.ndarray:
    """Blend the amounts of the two bands whose centres bracket each Oklab hue, flat on top of a band and a cosine between."""
    hue = np.where(hue < MIX_CENTRES[0], hue + np.float32(360.0), hue)
    band = np.minimum(np.searchsorted(_BAND_EDGES, hue, side="right") - 1, len(MIX_NAMES) - 1)
    lower = _BAND_EDGES[band]
    t = (hue - lower) / (_BAND_EDGES[band + 1] - lower)
    u = np.clip((t - np.float32(MIX_FLAT)) / np.float32(1.0 - 2.0 * MIX_FLAT), 0.0, 1.0)
    weight = (1.0 + np.cos(np.float32(math.pi) * u)) * np.float32(0.5)
    return weight * mix[band] + (1.0 - weight) * mix[(band + 1) % len(MIX_NAMES)]


def _boost_gain(chroma: np.ndarray) -> np.ndarray:
    """Return the boost gain: `MIX_BOOST_MUTED` on grey-ish colours, easing down to `MIX_BOOST_VIVID` at `MIX_VIVID_CHROMA`."""
    x = np.clip(chroma / np.float32(MIX_VIVID_CHROMA), 0.0, 1.0)
    return MIX_BOOST_VIVID + (MIX_BOOST_MUTED - MIX_BOOST_VIVID) * (1.0 - x) ** 2


def _gamut_factor(linear: np.ndarray, luma: np.ndarray) -> np.ndarray:
    """Return the largest saturation factor at which no channel of each pixel clips; at least 1e9 times the room for a grey."""
    d = linear - luma[:, None]
    peak = np.maximum(np.maximum(d[:, 0], d[:, 1]), d[:, 2])
    low = np.minimum(np.minimum(d[:, 0], d[:, 1]), d[:, 2])
    up = (1.0 - luma) / np.maximum(peak, np.float32(1e-9))
    down = luma / np.maximum(-low, np.float32(1e-9))
    return np.minimum(up, down)


def _contrast_curve(p: np.ndarray, contrast: int) -> np.ndarray:
    if not contrast:
        return p
    b = CONTRAST_MAX * abs(contrast) / 100.0
    centred = 2.0 * p - 1.0
    if contrast > 0:
        return 0.5 + 0.5 * np.tanh(b * centred) / math.tanh(b)
    return 0.5 + 0.5 * np.arctanh(centred * math.tanh(b)) / b


def _encode_table(contrast: int) -> np.ndarray:
    """Tabulate `contrast(encode(L))` at the points of an even grid over L in [0, 1]."""
    grid = np.arange(_ENCODE_INTERVALS + 1, dtype=np.float64) / _ENCODE_INTERVALS
    return _contrast_curve(_linear_to_srgb(grid), contrast).astype(np.float32)


def _channel_table(contrast: int) -> np.ndarray:
    """Tabulate `contrast(X / 255)` for every 8-bit value, for pixels that skip the linear-light stages."""
    return _contrast_curve(np.arange(256, dtype=np.float64) / 255.0, contrast).astype(np.float32)


def _lerp(table: np.ndarray, x: np.ndarray) -> np.ndarray:
    intervals = table.shape[0] - 1
    scaled = x * np.float32(intervals)
    index = np.minimum(scaled.astype(np.int32), intervals - 1)
    low = table[index]
    return low + (table[index + 1] - low) * (scaled - index)


def _tone(p: np.ndarray, highlights: float, shadows: float) -> np.ndarray:
    luma = p @ _LUMA
    toward_shadows = TONE_TOWARD if shadows > 0 else TONE_AWAY
    toward_highlights = TONE_TOWARD if highlights < 0 else TONE_AWAY
    inverse = 1.0 - luma
    inverse_squared = inverse * inverse
    luma_squared = luma * luma
    shift = luma * inverse * ((shadows * toward_shadows) * inverse_squared + (highlights * toward_highlights) * luma_squared)
    return np.clip(p + shift[:, None], 0.0, 1.0)
