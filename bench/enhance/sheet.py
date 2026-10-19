"""Render bench rewrites as images and lay them out side by side: one row per case, one column per run."""

from __future__ import annotations

import html
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from bench.enhance.report import Run

SHEET_FILE = "index.html"
PROMPTS_FILE = "prompts.yaml"


@dataclass(frozen=True)
class SheetCell:
    """One image to render: its prompt-file set name, the column it belongs to and the prompt text."""

    name: str
    column: str
    prompt: str


@dataclass(frozen=True)
class SheetRow:
    """One case: its id and the cells of every column, the user's own prompt first."""

    case: str
    cells: tuple[SheetCell, ...]


def sheet_rows(runs: tuple[Run, ...], cases: tuple[str, ...], *, input_column: str = "Your prompt") -> tuple[SheetRow, ...]:
    """Return one row per image case found in every run: the input prompt, then each run's rewrite (its first seed).

    Set names are short and safe (``s001`` …) because the image CLI turns a set name into the output file name.
    """
    rows: list[SheetRow] = []
    counter = 0
    for case in cases:
        records = [_first_record(run, case) for run in runs]
        if any(record is None or record["mode"] != "image" or record["error"] for record in records):
            continue
        cells = [(input_column, records[0]["input"]["prompt"])] + [(run.label, record["output"]) for run, record in zip(runs, records)]
        row_cells = []
        for column, prompt in cells:
            counter += 1
            row_cells.append(SheetCell(name=f"s{counter:03d}", column=column, prompt=prompt))
        rows.append(SheetRow(case=case, cells=tuple(row_cells)))
    return tuple(rows)


def _first_record(run: Run, case: str) -> dict[str, Any] | None:
    records = sorted((record for record in run.records if record["case"] == case), key=lambda record: record["seed"])
    return records[0] if records else None


def prompts_file_text(rows: tuple[SheetRow, ...]) -> str:
    """Return a prompt file with one set per cell, so one image-CLI run renders the whole sheet with one model load."""
    data = {cell.name: [{"prompt": cell.prompt}] for row in rows for cell in row.cells}
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=10_000)


def find_image(out_dir: Path, name: str) -> Path | None:
    """Return the image the CLI wrote for set *name* (``{name}_{timestamp}.png``), or None."""
    matches = sorted(out_dir.glob(f"{name}_*.png"))
    return matches[0] if matches else None


def render_command(prompts: Path, out_dir: Path, *, model: str, seed: int, ratio: str, size: str) -> list[str]:
    """Return the image-CLI command that renders every cell with one model, seed and size, enhancement off."""
    cli = Path(sys.executable).with_name("ziv-image")
    return [str(cli), "-m", model, "-p", str(prompts), "--seed", str(seed), "--ratio", ratio, "-s", size, "-o", str(out_dir), "--no-enhance"]


def sheet_html(rows: tuple[SheetRow, ...], out_dir: Path, *, title: str) -> str:
    """Return the sheet as a self-contained HTML page with relative image links."""
    columns = [cell.column for cell in rows[0].cells] if rows else []
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = []
        for cell in row.cells:
            image = find_image(out_dir, cell.name)
            picture = f'<a href="{image.name}"><img src="{image.name}" loading="lazy"></a>' if image else "<em>missing</em>"
            cells.append(f"<td>{picture}<p>{html.escape(cell.prompt)}</p></td>")
        body.append(f"<tr><th>{html.escape(row.case)}</th>{''.join(cells)}</tr>")
    style = (
        "body{background:#111214;color:#ececee;font:13px system-ui;margin:16px}table{border-collapse:collapse}"
        "th,td{vertical-align:top;padding:8px;border-bottom:1px solid #26282c}td{width:320px}img{width:320px;border-radius:6px}"
        "p{color:#b4b6bc;font-size:12px;line-height:1.4;margin:6px 0 0}th{text-align:left;color:#8f939a}"
    )
    return f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title><style>{style}</style><h1>{html.escape(title)}</h1><table><tr><th>Case</th>{head}</tr>{''.join(body)}</table>\n"


def build_sheet(runs: tuple[Run, ...], cases: tuple[str, ...], out_dir: Path, *, model: str, seed: int, ratio: str, size: str, echo=print) -> Path:
    """Render every cell with the image CLI into *out_dir* and write the HTML sheet; return its path.

    Raises:
        ValueError: When no case is an image case present in every run.
        subprocess.CalledProcessError: When the image CLI fails.
    """
    rows = sheet_rows(runs, cases)
    if not rows:
        raise ValueError("No image case is present in every run.")
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = out_dir / PROMPTS_FILE
    prompts.write_text(prompts_file_text(rows), encoding="utf-8")
    command = render_command(prompts, out_dir, model=model, seed=seed, ratio=ratio, size=size)
    echo(f"Rendering {sum(len(row.cells) for row in rows)} images with {model}…")
    subprocess.run(command, check=True)
    sheet = out_dir / SHEET_FILE
    sheet.write_text(sheet_html(rows, out_dir, title=f"{' vs '.join(run.label for run in runs)} · {model} · seed {seed}"), encoding="utf-8")
    return sheet
