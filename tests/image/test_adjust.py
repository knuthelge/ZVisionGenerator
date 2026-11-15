"""Unit tests for the Adjust editor's light maths against the shared fixture and a float64 reference."""

from __future__ import annotations

import json
import math
import random

import numpy as np
import pytest

import zvisiongenerator.processing.adjust as adjust_module
from tests._paths import FIXTURES
from zvisiongenerator.processing.adjust import MIX_CENTRES, MIX_NAMES, LightSettings, adjust_light

_KR, _KG, _KB = 0.2126, 0.7152, 0.0722
_MIX_SETTINGS = tuple(f"mix_{name}" for name in MIX_NAMES)
_SETTING_NAMES = ("exposure", "contrast", "highlights", "shadows", "saturation", "vibrance", "warmth", *_MIX_SETTINGS)

_ANCHORS = [
    ((128, 128, 128), {}, (128, 128, 128)),
    ((128, 128, 128), {"exposure": 50}, (176, 176, 176)),
    ((128, 128, 128), {"exposure": -50}, (92, 92, 92)),
    ((128, 128, 128), {"warmth": 100}, (146, 124, 106)),
    ((128, 128, 128), {"warmth": -100}, (111, 130, 152)),
    ((64, 64, 64), {"contrast": 100}, (38, 38, 38)),
    ((192, 192, 192), {"contrast": 100}, (218, 218, 218)),
    ((64, 64, 64), {"contrast": -100}, (86, 86, 86)),
    ((128, 128, 128), {"contrast": 100}, (128, 128, 128)),
    ((200, 40, 40), {"saturation": -100}, (104, 104, 104)),
    ((200, 40, 40), {"saturation": 100}, (255, 0, 0)),
    ((40, 40, 40), {"shadows": 100}, (76, 76, 76)),
    ((40, 40, 40), {"shadows": -100}, (16, 16, 16)),
    ((220, 220, 220), {"highlights": -100}, (186, 186, 186)),
    ((220, 220, 220), {"highlights": 100}, (242, 242, 242)),
    ((128, 128, 128), {"shadows": 100, "highlights": -100}, (128, 128, 128)),
    (
        (10, 220, 30),
        {"exposure": 30, "contrast": 20, "highlights": -40, "shadows": 25, "saturation": 15, "warmth": 10},
        (0, 241, 0),
    ),
]


def _lin(v: float) -> float:
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _enc(v: float) -> float:
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, v))


def _contrast(p: float, c: int) -> float:
    if c == 0:
        return p
    b = 1.5 * abs(c) / 100
    if c > 0:
        return 0.5 + 0.5 * math.tanh(b * (2 * p - 1)) / math.tanh(b)
    return 0.5 + 0.5 * math.atanh((2 * p - 1) * math.tanh(b)) / b


def _tone_shift(y: float, h: int, s: int) -> float:
    ks = 1.5 if s > 0 else 1.0
    kh = 1.5 if h < 0 else 1.0
    return (s / 100) * ks * y * (1 - y) ** 3 + (h / 100) * kh * y**3 * (1 - y)


def _vibrance_factor(s: float, saturation: int, vibrance: int) -> float:
    kv = 2.0 if vibrance > 0 else 1.0
    return (1 + saturation / 100) * (1 + kv * (vibrance / 100) * (1 - s) ** 2)


def _chroma(lin: list[float]) -> float:
    peak = max(lin)
    return (peak - min(lin)) / peak if peak > 0 else 0.0


def _oklab_hue_chroma(lin: list[float]) -> tuple[float, float]:
    """Return the Oklab hue in degrees (0 to 360) and the Oklab chroma of a linear sRGB colour."""
    r, g, b = lin
    lc = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    a = 1.9779984951 * lc - 2.4285922050 * m + 0.4505937099 * s
    bb = 0.0259040371 * lc + 0.7827717662 * m - 0.8086757660 * s
    return math.degrees(math.atan2(bb, a)) % 360, math.hypot(a, bb)


_FLAT = 0.25
_FADE = (0.002, 0.006)
_BOOST_MUTED, _BOOST_VIVID, _VIVID_CHROMA, _SOFTNESS = 3.0, 1.0, 0.2, 4.0


def _band_weights(hue: float) -> list[float]:
    """Return the eight band weights at an Oklab hue: flat on each centre, a raised cosine between; they sum to 1."""
    edges = [*MIX_CENTRES, MIX_CENTRES[0] + 360]
    if hue < edges[0]:
        hue += 360
    i = max(k for k in range(8) if edges[k] <= hue)
    t = (hue - edges[i]) / (edges[i + 1] - edges[i])
    u = _clamp01((t - _FLAT) / (1 - 2 * _FLAT))
    w = (1 + math.cos(math.pi * u)) / 2
    weights = [0.0] * 8
    weights[i] += w
    weights[(i + 1) % 8] += 1 - w
    return weights


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    u = _clamp01((x - edge0) / (edge1 - edge0))
    return u * u * (3 - 2 * u)


def _gamut_factor(lin: list[float], y: float) -> float:
    """Return the largest saturation factor before a channel clips; infinity for a grey."""
    limit = math.inf
    for v in lin:
        if v - y > 1e-9:
            limit = min(limit, (1 - y) / (v - y))
        elif v - y < -1e-9:
            limit = min(limit, y / (y - v))
    return limit


def _mixer_factor(lin: list[float], y: float, base: float, mix: dict[str, int]) -> float:
    """Return the saturation factor after the colour mixer changed `base`, the saturation and vibrance factor."""
    amounts = [mix.get(name, 0) / 100 for name in _MIX_SETTINGS]
    if not any(amounts):
        return base
    hue, chroma = _oklab_hue_chroma(lin)
    if chroma <= 1e-6:
        return base
    amount = sum(w * a for w, a in zip(_band_weights(hue), amounts, strict=True))
    if amount <= 0:
        return base * (1 + amount)
    gain = _BOOST_VIVID + (_BOOST_MUTED - _BOOST_VIVID) * (1 - min(chroma / _VIVID_CHROMA, 1)) ** 2
    excess = amount * gain * _smoothstep(*_FADE, chroma) * base
    headroom = max(_gamut_factor(lin, y) - base, 0.0)
    if excess <= 0:
        return base
    if math.isinf(headroom):
        return base + excess
    return base + excess * headroom / (excess**_SOFTNESS + headroom**_SOFTNESS) ** (1 / _SOFTNESS)


def _mix(lin: list[float], y: float, saturation: int, vibrance: int, mix: dict[str, int] | None = None) -> list[float]:
    """Apply step 3, saturation, vibrance and the colour mixer in linear light, each measure read before any applies."""
    f = _mixer_factor(lin, y, _vibrance_factor(_chroma(lin), saturation, vibrance), mix or {})
    return [_clamp01(y + f * (v - y)) for v in lin]


def _reference_unrounded(pixel: tuple[int, int, int], s: dict[str, int]) -> list[float]:
    """Run the float64 reference up to, but not including, the final quantization."""
    k = s.get("warmth", 0) / 100
    warm = (2 ** (0.5 * k), 1.0, 2 ** (-0.5 * k))
    n = _KR * warm[0] + _KG * warm[1] + _KB * warm[2]
    x = 2 ** (2 * s.get("exposure", 0) / 100)
    lin = [_clamp01(_lin(v / 255) * x * g / n) for v, g in zip(pixel, warm, strict=True)]
    y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
    lin = _mix(lin, y, s.get("saturation", 0), s.get("vibrance", 0), s)
    p = [_contrast(_enc(v), s.get("contrast", 0)) for v in lin]
    luma = _KR * p[0] + _KG * p[1] + _KB * p[2]
    shift = _tone_shift(luma, s.get("highlights", 0), s.get("shadows", 0))
    return [_clamp01(v + shift) for v in p]


def _reference(pixel: tuple[int, int, int], settings: dict[str, int]) -> list[int]:
    return [math.floor(v * 255 + 0.5) for v in _reference_unrounded(pixel, settings)]


def _run(pixels: list[list[int]], settings: dict[str, int]) -> np.ndarray:
    rgb = np.array([pixels], dtype=np.uint8)
    return adjust_light(rgb, LightSettings(**settings))[0]


def _load_cases() -> list[dict]:
    return json.loads((FIXTURES / "adjust_cases.json").read_text())


_CASES = _load_cases()


class TestFixture:
    def test_has_the_light_and_colour_mixer_cases(self):
        assert len(_CASES) >= 119

    @pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
    def test_expected_comes_from_the_reference(self, case):
        assert set(case["settings"]) <= set(_SETTING_NAMES)
        assert len(case["pixels"]) == len(case["expected"])
        for pixel, expected in zip(case["pixels"], case["expected"], strict=True):
            assert _reference(tuple(pixel), case["settings"]) == expected

    @pytest.mark.parametrize(("pixel", "settings", "expected"), _ANCHORS)
    def test_anchors_are_listed_and_reproduce_exactly(self, pixel, settings, expected):
        assert tuple(_reference(pixel, settings)) == expected
        assert any(c["settings"] == settings and pixel in map(tuple, c["pixels"]) and expected in map(tuple, c["expected"]) for c in _CASES)
        assert tuple(_run([list(pixel)], settings)[0]) == expected


class TestParity:
    @pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
    def test_adjust_light_is_within_one_of_the_fixture(self, case):
        out = _run(case["pixels"], case["settings"]).astype(int)
        assert np.abs(out - np.array(case["expected"])).max() <= 1

    def test_random_settings_and_pixels_are_within_one_of_the_reference(self):
        rng = random.Random(7)
        for _ in range(40):
            settings = {name: rng.choice([0, rng.randint(-100, 100)]) for name in _SETTING_NAMES}
            pixels = [[rng.randrange(256) for _ in range(3)] for _ in range(60)]
            expected = np.array([_reference(tuple(p), settings) for p in pixels])
            assert np.abs(_run(pixels, settings).astype(int) - expected).max() <= 1, settings

    def test_an_image_larger_than_one_chunk_matches_the_reference(self):
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, size=(1100, 1000, 3), dtype=np.uint8)
        settings = {"exposure": 20, "saturation": 30, "shadows": 10}
        out = adjust_light(rgb, LightSettings(**settings))
        for y, x in ((0, 0), (1048, 575), (1049, 576), (1099, 999)):
            expected = _reference(tuple(int(v) for v in rgb[y, x]), settings)
            assert np.abs(out[y, x].astype(int) - expected).max() <= 1


def _vibrance_cases() -> list[tuple[str, dict[str, int], tuple[int, int, int], tuple[int, int, int]]]:
    rows = [
        ((150, 130, 120), {-100: (144, 132, 126), -50: (147, 131, 123), 50: (156, 128, 114), 100: (161, 127, 108)}),
        ((224, 172, 140), {-100: (219, 173, 146), -50: (222, 173, 143), 50: (228, 171, 133), 100: (233, 169, 126)}),
        ((200, 40, 40), dict.fromkeys((-100, -50, 50, 100), (200, 40, 40))),
        ((128, 128, 128), dict.fromkeys((-100, -50, 50, 100), (128, 128, 128))),
    ]
    cases = [(f"{pixel} v{v:+d}", {"vibrance": v}, pixel, out) for pixel, outs in rows for v, out in outs.items()]
    cases += [
        ("muted t+30 v+50", {"saturation": 30, "vibrance": 50}, (150, 130, 120), (161, 127, 107)),
        ("muted t-50 v+100", {"saturation": -50, "vibrance": 100}, (150, 130, 120), (148, 130, 122)),
        ("muted t-100 v+100", {"saturation": -100, "vibrance": 100}, (150, 130, 120), (134, 134, 134)),
        ("skin e+30 w+20 v+60", {"exposure": 30, "warmth": 20, "vibrance": 60}, (224, 172, 140), (255, 205, 152)),
    ]
    return cases


class TestVibranceAnchors:
    @pytest.mark.parametrize(("name", "settings", "pixel", "expected"), _vibrance_cases(), ids=lambda v: v if isinstance(v, str) else "")
    def test_the_reference_reproduces_the_anchor_exactly_and_adjust_light_is_within_one(self, name, settings, pixel, expected):
        assert tuple(_reference(pixel, settings)) == expected
        assert np.abs(_run([list(pixel)], settings)[0].astype(int) - expected).max() <= 1
        assert any(c["settings"] == settings and list(pixel) in c["pixels"] and list(expected) in c["expected"] for c in _CASES)


class TestShape:
    def test_neutral_settings_return_an_equal_copy(self):
        rgb = np.random.default_rng(1).integers(0, 256, size=(7, 9, 3), dtype=np.uint8)
        out = adjust_light(rgb, LightSettings())
        assert np.array_equal(out, rgb)
        assert out is not rgb
        assert not np.shares_memory(out, rgb)

    def test_is_neutral_reports_any_nonzero_setting(self):
        assert LightSettings().is_neutral
        assert not LightSettings(warmth=1).is_neutral

    def test_the_input_is_not_modified(self):
        rgb = np.full((2, 2, 3), 100, dtype=np.uint8)
        adjust_light(rgb, LightSettings(exposure=50, contrast=30))
        assert (rgb == 100).all()

    def test_an_empty_image_is_returned_empty(self):
        out = adjust_light(np.zeros((0, 5, 3), dtype=np.uint8), LightSettings(exposure=10))
        assert out.shape == (0, 5, 3)

    @pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
    def test_other_shapes_raise(self, shape):
        with pytest.raises(ValueError, match="shape"):
            adjust_light(np.zeros(shape, dtype=np.uint8), LightSettings(exposure=1))

    def test_other_dtypes_raise(self):
        with pytest.raises(ValueError, match="uint8"):
            adjust_light(np.zeros((2, 2, 3), dtype=np.float32), LightSettings(exposure=1))

    @pytest.mark.parametrize("name", _SETTING_NAMES)
    def test_settings_outside_the_range_raise(self, name):
        with pytest.raises(ValueError, match=name):
            LightSettings(**{name: 101})
        with pytest.raises(ValueError, match=name):
            LightSettings(**{name: -101})


class TestInvariants:
    def test_the_contrast_pivot_is_fixed(self):
        for c in (-100, -40, 40, 100):
            assert _contrast(0.5, c) == pytest.approx(0.5, abs=1e-12)
            assert tuple(_run([[128, 128, 128]], {"contrast": c})[0]) == (128, 128, 128)

    @pytest.mark.parametrize("amount", [20, 60, 100])
    def test_contrast_then_its_negative_returns_the_input_within_one(self, amount):
        grey = np.arange(256, dtype=np.uint8)
        rgb = np.stack([grey, grey, grey], axis=-1)[None]
        for first, second in ((amount, -amount), (-amount, amount)):
            there = adjust_light(rgb, LightSettings(contrast=first))
            back = adjust_light(there, LightSettings(contrast=second))
            assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 1

    def test_contrast_negative_is_the_exact_inverse_in_float(self):
        for c in (10, 55, 100):
            for p in np.linspace(0, 1, 101):
                assert _contrast(_contrast(float(p), c), -c) == pytest.approx(p, abs=1e-9)

    def test_the_tone_curve_never_slopes_down(self):
        grid = np.linspace(0, 1, 1001)
        for s in (-100, -50, 0, 50, 100):
            for h in (-100, -50, 0, 50, 100):
                curve = np.array([y + _tone_shift(float(y), h, s) for y in grid])
                slope = np.diff(curve) * 1000
                assert slope.min() >= -1e-9, (s, h)
                assert curve[0] == 0 and curve[-1] == pytest.approx(1)

    def test_the_tone_curve_minimum_slope_is_a_quarter_at_full_strength_toward_mid_grey(self):
        grid = np.linspace(0, 1, 1001)
        curve = np.array([y + _tone_shift(float(y), -100, 100) for y in grid])
        assert (np.diff(curve) * 1000).min() == pytest.approx(0.25, abs=1e-3)

    @pytest.mark.parametrize("grey", [10, 80, 128, 200])
    def test_warmth_keeps_a_greys_luminance(self, grey):
        for warmth in (-100, -40, 40, 100):
            r, g, b = _run([[grey] * 3], {"warmth": warmth})[0].astype(float)
            luma_in = _lin(grey / 255)
            luma_out = _KR * _lin(r / 255) + _KG * _lin(g / 255) + _KB * _lin(b / 255)
            assert luma_out == pytest.approx(luma_in, abs=0.02 * max(luma_in, 0.01) + 0.002)

    def test_full_desaturation_gives_equal_channels(self):
        rng = np.random.default_rng(5)
        rgb = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
        out = adjust_light(rgb, LightSettings(saturation=-100))
        assert (out[..., 0] == out[..., 1]).all()
        assert (out[..., 1] == out[..., 2]).all()


_HUES = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (0, 1, 1), (1, 0, 1), (1, 0.5, 0.2), (0.3, 0.2, 1))
_VIBRANCES = (-100, -50, 50, 100)


def _hue_line(hue: tuple[float, float, float], y: float, steps: int = 400) -> list[list[float]]:
    """Return linear colours of luminance *y* along *hue*, from grey to the edge of the gamut."""
    direction = [h - (_KR * hue[0] + _KG * hue[1] + _KB * hue[2]) for h in hue]
    reach = min((1 - y) / d if d > 0 else y / -d for d in direction if d)
    return [[y + reach * i / steps * d for d in direction] for i in range(steps + 1)]


class TestVibranceProperties:
    @pytest.mark.parametrize("vibrance", _VIBRANCES)
    def test_luminance_is_unchanged_on_unclipped_pixels(self, vibrance):
        rng = random.Random(11)
        checked = 0
        for _ in range(2000):
            lin = [rng.uniform(0.05, 0.95) for _ in range(3)]
            y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
            out = _mix(lin, y, 0, vibrance)
            if all(0.0 < v < 1.0 for v in out):
                checked += 1
                assert _KR * out[0] + _KG * out[1] + _KB * out[2] == pytest.approx(y, abs=1e-6)
        assert checked > 500

    @pytest.mark.parametrize("vibrance", _VIBRANCES)
    @pytest.mark.parametrize("grey", [0, 1, 40, 128, 254, 255])
    def test_greys_are_unchanged(self, grey, vibrance):
        assert tuple(_run([[grey] * 3], {"vibrance": vibrance})[0]) == (grey, grey, grey)

    @pytest.mark.parametrize("vibrance", _VIBRANCES)
    @pytest.mark.parametrize("y", [0.05, 0.2, 0.4, 0.6])
    @pytest.mark.parametrize("hue", _HUES)
    def test_output_chroma_never_falls_as_input_chroma_grows(self, hue, y, vibrance):
        line = _hue_line(hue, y)
        chroma = []
        for lin in line:
            out = _mix(lin, y, 0, vibrance)
            chroma.append(max(out) - min(out))
        assert np.diff(chroma).min() >= -1e-12

    @pytest.mark.parametrize("vibrance", [-100, -37, 0, 25, 100])
    def test_full_desaturation_gives_equal_channels_whatever_the_vibrance(self, vibrance):
        rng = np.random.default_rng(9)
        rgb = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
        out = adjust_light(rgb, LightSettings(saturation=-100, vibrance=vibrance))
        assert (out[..., 0] == out[..., 1]).all()
        assert (out[..., 1] == out[..., 2]).all()

    def test_vibrance_zero_is_exactly_the_saturation_step(self):
        rng = np.random.default_rng(4)
        rgb = rng.integers(0, 256, size=(30, 30, 3), dtype=np.uint8)
        assert np.array_equal(adjust_light(rgb, LightSettings(saturation=40, vibrance=0)), adjust_light(rgb, LightSettings(saturation=40)))

    def test_vibrance_alone_is_not_neutral(self):
        assert not LightSettings(vibrance=1).is_neutral


_MIXER_TABLE = {
    "red": ((184, 92, 80), (218, 63, 7), (119, 119, 119), (176, 88, 52), (188, 80, 1), (149, 101, 89)),
    "orange": ((196, 128, 80), (213, 121, 0), (144, 144, 144), (180, 136, 40), (183, 135, 0), (163, 140, 108)),
    "yellow": ((156, 156, 44), (156, 156, 0), (151, 151, 151), (160, 188, 116), (148, 193, 37), (169, 183, 151)),
    "green": ((116, 184, 108), (58, 193, 7), (168, 168, 168), (72, 184, 148), (0, 188, 144), (129, 175, 157)),
    "aqua": ((56, 184, 184), (0, 186, 186), (167, 167, 167), (44, 144, 184), (0, 145, 189), (102, 139, 162)),
    "blue": ((80, 112, 180), (12, 111, 220), (113, 113, 113), (96, 100, 184), (87, 95, 215), (102, 104, 153)),
    "purple": ((136, 120, 188), (144, 105, 241), (130, 130, 130), (160, 124, 184), (174, 112, 210), (150, 131, 164)),
    "magenta": ((184, 112, 180), (225, 67, 218), (137, 137, 137), (180, 100, 128), (209, 77, 130), (155, 113, 126)),
}


def _mixer_anchors() -> list[tuple[str, dict[str, int], tuple[int, int, int], tuple[int, int, int]]]:
    cases = []
    for name, (centre, centre_up, centre_down, half, half_up, half_down) in _MIXER_TABLE.items():
        for where, pixel, up, down in (("centre", centre, centre_up, centre_down), ("halfway", half, half_up, half_down)):
            cases.append((f"{name} {where} +100", {f"mix_{name}": 100}, pixel, up))
            cases.append((f"{name} {where} -100", {f"mix_{name}": -100}, pixel, down))
        cases.append((f"{name} on grey", {f"mix_{name}": 100}, (128, 128, 128), (128, 128, 128)))
    cases += [
        ("muted lake blue +100", {"mix_blue": 100}, (92, 104, 124), (57, 106, 160)),
        ("pale sky +100", {"mix_blue": 100}, (120, 126, 134), (104, 128, 155)),
        ("near-grey is not tinted", {"mix_red": 100}, (129, 128, 127), (129, 128, 127)),
        ("mixed orange -40 red +25", {"saturation": 20, "vibrance": 30, "mix_orange": -40, "mix_red": 25}, (196, 128, 80), (184, 132, 102)),
        ("mixed vibrance with orange", {"vibrance": 50, "mix_orange": 50}, (196, 128, 80), (212, 121, 4)),
        ("black and white wins", {"saturation": -100, "mix_red": 100}, (184, 92, 80), (119, 119, 119)),
    ]
    return cases


def _oklab_to_linear(lightness: float, chroma: float, hue: float) -> list[float]:
    a, b = chroma * math.cos(math.radians(hue)), chroma * math.sin(math.radians(hue))
    lc = (lightness + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return [
        4.0767416621 * lc - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * lc + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * lc - 0.7034186147 * m + 1.7076147010 * s,
    ]


class TestColourMixerAnchors:
    def test_the_table_has_forty_six_anchors(self):
        assert len(_mixer_anchors()) == 46

    @pytest.mark.parametrize(("name", "settings", "pixel", "expected"), _mixer_anchors(), ids=lambda v: v if isinstance(v, str) else "")
    def test_the_reference_reproduces_the_anchor_exactly_and_adjust_light_is_within_one(self, name, settings, pixel, expected):
        assert tuple(_reference(pixel, settings)) == expected
        assert np.abs(_run([list(pixel)], settings)[0].astype(int) - expected).max() <= 1
        assert any(c["settings"] == settings and list(pixel) in c["pixels"] and list(expected) in c["expected"] for c in _CASES)

    def test_the_band_centres_are_the_oklab_hues_of_the_named_srgb_colours_to_within_two_degrees(self):
        named = ((255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255))
        for pixel, centre in zip(named, MIX_CENTRES, strict=True):
            hue, _ = _oklab_hue_chroma([_lin(v / 255) for v in pixel])
            assert abs(hue - centre) <= 2.1


class TestColourMixerProperties:
    def test_the_weights_sum_to_one_at_every_hue_with_no_seam(self):
        hues = np.arange(36000) / 100
        weights = np.array([_band_weights(float(h)) for h in hues])
        assert np.abs(weights.sum(axis=1) - 1).max() < 1e-12
        assert (weights >= 0).all()
        ring = np.vstack([weights, weights[:1]])
        assert np.abs(np.diff(ring, axis=0)).max() < 1.5e-3

    def test_a_band_has_full_weight_on_its_own_centre_and_none_on_the_others(self):
        for i, centre in enumerate(MIX_CENTRES):
            weights = _band_weights(float(centre))
            assert weights[i] == pytest.approx(1.0)
            assert sum(weights) - weights[i] == pytest.approx(0.0, abs=1e-12)

    def test_a_band_keeps_full_weight_for_a_quarter_of_the_gap_on_each_side_of_its_centre(self):
        edges = [*MIX_CENTRES, MIX_CENTRES[0] + 360]
        for i, centre in enumerate(MIX_CENTRES):
            reach_up = 0.25 * (edges[i + 1] - centre)
            reach_down = 0.25 * (centre - edges[i - 1] + (360 if i == 0 else 0))
            for hue in (centre - reach_down, centre - reach_down / 2, centre + reach_up / 2, centre + reach_up):
                assert _band_weights(float(hue))[i] == pytest.approx(1.0, abs=1e-9)
            assert _band_weights(centre + 1.5 * reach_up)[i] < 1.0 - 1e-3

    def test_the_weights_are_smooth_where_the_flat_top_meets_the_cosine(self):
        edges = [*MIX_CENTRES, MIX_CENTRES[0] + 360]
        for i, centre in enumerate(MIX_CENTRES):
            knee = centre + 0.25 * (edges[i + 1] - centre)
            below, above = (_band_weights(knee + d)[i] for d in (-1e-4, 1e-4))
            assert (1 - below) == pytest.approx(1 - above, abs=1e-7)

    @pytest.mark.parametrize("name", _MIX_SETTINGS)
    @pytest.mark.parametrize("amount", [-100, -40, 60, 100])
    def test_luminance_is_unchanged_on_unclipped_pixels(self, name, amount):
        rng = random.Random(21)
        checked = 0
        for _ in range(3000):
            lin = [rng.uniform(0.05, 0.95) for _ in range(3)]
            y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
            out = _mix(lin, y, 0, 0, {name: amount})
            if all(0.0 < v < 1.0 for v in out):
                checked += 1
                assert _KR * out[0] + _KG * out[1] + _KB * out[2] == pytest.approx(y, abs=1e-6)
        assert checked > 500

    def test_the_direction_from_luminance_is_kept_until_a_channel_clips(self):
        rng = random.Random(22)
        for _ in range(500):
            lin = [rng.uniform(0.1, 0.9) for _ in range(3)]
            y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
            out = _mix(lin, y, 0, 0, {"mix_orange": 70, "mix_blue": -50, "mix_green": 90})
            if all(0.0 < v < 1.0 for v in out):
                offsets = [(o - y) / (v - y) for o, v in zip(out, lin, strict=True) if abs(v - y) > 1e-3]
                assert max(offsets) - min(offsets) < 1e-6

    @pytest.mark.parametrize("name", _MIX_SETTINGS)
    @pytest.mark.parametrize("grey", [0, 1, 40, 128, 254, 255])
    def test_greys_are_unchanged_by_each_band(self, name, grey):
        for amount in (-100, 100):
            assert tuple(_run([[grey] * 3], {name: amount})[0]) == (grey, grey, grey)

    def test_greys_with_noise_below_the_fade_are_not_tinted_by_boosting(self):
        rng = random.Random(23)
        pixels = []
        while len(pixels) < 3000:
            grey = rng.randrange(256)
            pixel = [min(255, max(0, grey + rng.randint(-1, 1))) for _ in range(3)]
            _, chroma = _oklab_hue_chroma([_lin(v / 255) for v in pixel])
            if chroma < _FADE[0]:
                pixels.append(pixel)
        boosted = dict.fromkeys(_MIX_SETTINGS, 100)
        assert np.array_equal(_run(pixels, boosted), np.array(pixels))

    def test_a_boost_never_clips_or_shifts_the_hue_of_a_colour(self):
        rng = random.Random(24)
        for _ in range(2000):
            lin = [_lin(rng.randrange(256) / 255) for _ in range(3)]
            y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
            settings = {name: rng.randint(0, 100) for name in _MIX_SETTINGS}
            out = _mix(lin, y, 0, 0, settings)
            scales = [(o - y) / (v - y) for o, v in zip(out, lin, strict=True) if abs(v - y) > 1e-3]
            assert max(scales) - min(scales) < 1e-6, (lin, settings)

    def test_a_stronger_boost_never_lowers_a_colours_saturation(self):
        rng = random.Random(25)
        for _ in range(500):
            lin = [_lin(rng.randrange(256) / 255) for _ in range(3)]
            y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
            name = rng.choice(_MIX_SETTINGS)
            spread = [max(out) - min(out) for out in (_mix(lin, y, 0, 0, {name: amount}) for amount in range(0, 101, 10))]
            assert np.diff(spread).min() >= -1e-9, (lin, name)

    def test_boosting_pushes_a_muted_colour_further_than_the_gamut_would_allow_a_vivid_one(self):
        muted = _run([[92, 104, 124]], {"mix_blue": 100})[0].astype(int)
        assert muted[2] - muted[0] > 124 - 92 + 30

    @pytest.mark.parametrize("band", range(8))
    def test_a_band_leaves_a_colour_on_another_bands_centre_unchanged(self, band):
        for other, centre in enumerate(MIX_CENTRES):
            if other == band:
                continue
            lin = _oklab_to_linear(0.65, 0.08, float(centre))
            y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
            for amount in (-100, 100):
                out = _mix(lin, y, 0, 0, {_MIX_SETTINGS[band]: amount})
                assert out == pytest.approx(lin, abs=1e-12) or abs(MIX_CENTRES[band] - centre) % 360 == 0

    def test_a_band_leaves_the_other_bands_centre_colours_within_two(self):
        for band, name in enumerate(MIX_NAMES):
            for other, (centre, *_) in enumerate(_MIXER_TABLE.values()):
                if other == band:
                    continue
                for amount in (-100, 100):
                    out = _run([list(centre)], {f"mix_{name}": amount})[0].astype(int)
                    # The 8-bit colours sit within about a degree of their centre, so a far band's share is tiny.
                    if min(abs(other - band), 8 - abs(other - band)) > 1:
                        assert np.abs(out - np.array(centre)).max() <= 1, (name, other, amount)

    @pytest.mark.parametrize("name", _MIX_SETTINGS)
    @pytest.mark.parametrize("lightness", [0.3, 0.5, 0.7, 0.85])
    def test_muting_never_lowers_the_output_chroma_as_input_chroma_grows_along_a_hue_line(self, lightness, name):
        """A hue line keeps the Oklab hue and lightness and grows the Oklab chroma, as far as the sRGB gamut allows."""
        centres = (*MIX_CENTRES, *(c + 13 for c in MIX_CENTRES))
        for hue in centres:
            chroma = []
            for step in range(401):
                lin = _oklab_to_linear(lightness, 0.5 * step / 400, float(hue))
                if not all(0.0 <= v <= 1.0 for v in lin):
                    break
                y = _KR * lin[0] + _KG * lin[1] + _KB * lin[2]
                out = _mix(lin, y, 0, 0, {name: -100})
                chroma.append(max(out) - min(out))
            assert len(chroma) > 3
            assert np.diff(chroma).min() >= -1e-9, (hue, name)

    @pytest.mark.parametrize("amount", [-100, 100])
    @pytest.mark.parametrize("luminance", [0.05, 0.2, 0.4, 0.7])
    @pytest.mark.parametrize("direction", [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1), (0.5, 0, 1), (1, 0, 1), (0.7, 0.4, 0.1)])
    def test_output_chroma_never_falls_as_chroma_grows_along_a_line_of_constant_luminance(self, luminance, direction, amount):
        """With every band at the same amount, the factor depends on chroma only; the line is straight in linear light."""
        d = np.array(direction, dtype=float) - np.array(direction) @ np.array([_KR, _KG, _KB])
        spread = []
        for c in np.linspace(0.0005, 2, 1500):
            lin = list(luminance + c * d)
            if min(lin) < 0 or max(lin) > 1:
                break
            out = _mix(lin, luminance, 0, 0, dict.fromkeys(_MIX_SETTINGS, amount))
            spread.append(math.dist(out, [luminance] * 3))
        assert len(spread) < 3 or np.diff(spread).min() >= -1e-9

    def test_full_desaturation_gives_equal_channels_with_any_mix(self):
        rng = np.random.default_rng(31)
        rgb = rng.integers(0, 256, size=(30, 30, 3), dtype=np.uint8)
        for amounts in ({"mix_red": 100}, {"mix_blue": -100, "mix_green": 100, "mix_magenta": 50}, dict.fromkeys(_MIX_SETTINGS, 100)):
            out = adjust_light(rgb, LightSettings(saturation=-100, **amounts))
            assert (out[..., 0] == out[..., 1]).all()
            assert (out[..., 1] == out[..., 2]).all()

    def test_a_mix_matches_the_reference_all_around_the_hue_circle(self):
        pixels = [[min(255, max(0, round(v * 255))) for v in (_enc(_clamp01(c)) for c in _oklab_to_linear(0.65, 0.07, h / 4))] for h in range(1440)]
        rng = random.Random(41)
        for _ in range(6):
            settings = {name: rng.randint(-100, 100) for name in _MIX_SETTINGS}
            expected = np.array([_reference(tuple(p), settings) for p in pixels])
            assert np.abs(_run(pixels, settings).astype(int) - expected).max() <= 1, settings

    def test_a_mix_with_other_settings_matches_the_reference_on_random_pixels(self):
        rng = random.Random(42)
        for _ in range(40):
            settings = {name: rng.choice([0, rng.randint(-100, 100)]) for name in _SETTING_NAMES}
            settings[rng.choice(_MIX_SETTINGS)] = rng.choice([-100, -30, 30, 100])
            pixels = [[rng.randrange(256) for _ in range(3)] for _ in range(60)]
            expected = np.array([_reference(tuple(p), settings) for p in pixels])
            assert np.abs(_run(pixels, settings).astype(int) - expected).max() <= 1, settings

    def test_an_image_larger_than_one_chunk_applies_the_mix_to_every_chunk(self):
        rng = np.random.default_rng(32)
        rgb = rng.integers(0, 256, size=(1100, 1000, 3), dtype=np.uint8)
        settings = {"mix_orange": 80, "mix_blue": -60, "vibrance": 20}
        out = adjust_light(rgb, LightSettings(**settings))
        for y, x in ((0, 0), (1048, 575), (1049, 576), (1099, 999)):
            expected = _reference(tuple(int(v) for v in rgb[y, x]), settings)
            assert np.abs(out[y, x].astype(int) - expected).max() <= 1


class TestColourMixerShape:
    def test_a_mix_alone_is_not_neutral_and_changes_a_coloured_pixel(self):
        assert LightSettings().is_neutral
        assert not LightSettings(mix_aqua=-1).is_neutral
        assert tuple(_run([[184, 92, 80]], {"mix_red": 100})[0]) != (184, 92, 80)

    def test_the_mix_amounts_follow_the_band_order(self):
        light = LightSettings(**{name: i + 1 for i, name in enumerate(_MIX_SETTINGS)})
        assert light.mix == (1, 2, 3, 4, 5, 6, 7, 8)

    def test_the_mixer_is_skipped_when_every_band_is_zero(self, monkeypatch):
        def refuse(*_args, **_kwargs):
            raise AssertionError("the colour mixer ran with every band at 0")

        monkeypatch.setattr(adjust_module, "_mix_factor", refuse)
        rgb = np.random.default_rng(33).integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
        adjust_light(rgb, LightSettings(saturation=20, vibrance=30, exposure=10))

    def test_the_mixer_runs_when_a_band_is_set(self, monkeypatch):
        calls = []
        original = adjust_module._mix_factor
        monkeypatch.setattr(adjust_module, "_mix_factor", lambda *a: calls.append(1) or original(*a))
        adjust_light(np.full((2, 2, 3), 90, dtype=np.uint8), LightSettings(mix_green=5))
        assert calls

    def test_a_mix_with_saturation_and_vibrance_at_zero_is_not_the_same_as_none(self):
        rng = np.random.default_rng(34)
        rgb = rng.integers(0, 256, size=(20, 20, 3), dtype=np.uint8)
        assert not np.array_equal(adjust_light(rgb, LightSettings(mix_red=100)), rgb)
