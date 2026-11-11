"""Image-edit options of ``ziv-image``: ``--edit``, ``--mask`` and ``--extend``, and the refusals that keep an edit apart from a generation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from zvisiongenerator.utils.edit_request import check_edit_inputs, edit_mode, load_paint, parse_extend, source_is_cutout
from zvisiongenerator.utils.input_images import upright_size

__all__ = ["add_edit_arguments", "check_edit_source", "edit_output_folder", "refuse_edit_options"]


def add_edit_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--edit``, ``--mask`` and ``--extend`` to the image parser."""
    parser.add_argument(
        "--edit",
        dest="edit_source",
        type=str,
        default=None,
        metavar="IMAGE",
        help="Edit this picture with --prompt instead of generating one: redraw the area painted in --mask, grow a side with --extend, or, with neither, restyle the whole picture (FLUX.2 Klein, opaque pictures). "
        "Saved as IMAGE's name plus _edit.png beside it unless -o is set.",
    )
    parser.add_argument(
        "--mask",
        dest="edit_mask",
        type=str,
        default=None,
        metavar="PNG",
        help="The area of the --edit picture to redraw: a PNG of the picture's shape (any size of it), painted where the picture should change: any pixel that is not fully transparent when the PNG has transparency, otherwise any pixel that is not black. Needs --edit.",
    )
    parser.add_argument(
        "--extend",
        dest="edit_extend",
        type=str,
        default=None,
        metavar="SIDE=PIXELS",
        help="Grow the --edit picture on one side (top, right, bottom or left) by 128 to 512 pixels in steps of 16, for example top=256, and draw the new area. Needs --edit.",
    )


def refuse_edit_options(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse the option combinations an edit cannot honour, and turn the edit's own options into the arguments the batch reads.

    Runs before any model is detected or loaded. Without ``--edit`` it only refuses ``--mask`` and ``--extend``. With it, it
    sets ``edit_extend`` to a ``(side, amount)`` pair and turns off the work that belongs to a generation: enhancement,
    upscaling, cut-out, references, a start image and sharpening.
    """
    if args.edit_source is None:
        for option, value in (("--mask", args.edit_mask), ("--extend", args.edit_extend)):
            if value is not None:
                parser.error(f"{option} needs --edit: it says what to change in the picture being edited.")
        return
    if args.prompt is None and args.json_prompt is None:
        parser.error("--edit needs --prompt: say what the picture should become.")
    given = (
        ("--image", args.image_path is not None),
        ("--reference", bool(args.reference_paths)),
        ("--upscale", args.upscale is not None),
        ("--review", args.review),
        ("--width", args.width is not None),
        ("--height", args.height is not None),
        ("--ratio", args.ratio is not None),
        ("--size", args.size is not None),
        ("--guidance", args.guidance is not None),
        ("--scheduler", args.scheduler is not None),
        ("--json-prompt", args.json_prompt is not None),
        ("--enhance", args.enhance is not None),
        ("--enhance-note", args.enhance_note is not None),
        ("--enhance-format", args.enhance_format is not None),
        ("--transparent", args.transparent),
        ("--contrast", args.contrast is not False),
        ("--saturation", args.saturation is not False),
        ("--sharpen AMOUNT", isinstance(args.sharpen, float)),
    )
    for option, is_given in given:
        if is_given:
            parser.error(f"{option} cannot be combined with --edit: an edit keeps the picture's own size and look.")
    if args.edit_extend is not None:
        try:
            args.edit_extend = parse_extend(args.edit_extend)
        except ValueError as error:
            parser.error(f"--extend: {error}")
    args.no_enhance = True
    args.sharpen = False


def check_edit_source(parser: argparse.ArgumentParser, args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    """Refuse an edit the model or the files cannot do, before the model loads, and read the mask into ``args.edit_paint``.

    Args:
        parser: The parser that reports the refusal.
        args: Parsed arguments with ``edit_source``, ``edit_mask`` and ``edit_extend`` set and their paths expanded.
        defaults: The model's resolved settings, which carry ``supports_mask_edit`` and ``max_reference_images``.
    """
    source = args.edit_source
    if not os.path.isfile(source):
        parser.error(f"Image to edit not found: {source}")
    try:
        width, height = upright_size(source)
        cutout = source_is_cutout(source)
    except (OSError, ValueError) as error:
        parser.error(f"The image to edit is not a readable picture: {error}")
    mode = edit_mode(args.edit_mask is not None, args.edit_extend)
    try:
        check_edit_inputs(mode, args.edit_extend, cutout, defaults.get("max_reference_images", 0), bool(defaults.get("supports_mask_edit", False)))
        args.edit_paint = load_paint(args.edit_mask, width, height) if args.edit_mask is not None else None
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


def edit_output_folder(args: argparse.Namespace) -> str:
    """Return the folder results go to: ``-o`` when set, else the edited picture's own folder, else the current folder."""
    if args.output:
        return args.output
    return str(Path(args.edit_source).parent) if args.edit_source else "."
