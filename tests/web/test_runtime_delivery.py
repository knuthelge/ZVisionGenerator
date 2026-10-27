from __future__ import annotations

from email.parser import BytesParser
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest

from tests._paths import REPO_ROOT

from zvisiongenerator.web.workspace_contract import build_workflow_contract, canonicalize_workflow, workflow_mode


LICENSE_FILES = ("LICENSE", "THIRD_PARTY_LICENSES.md")
LICENSE_EXPRESSION = "AGPL-3.0-or-later"
VENDORED_PACKAGES = ("ltx_core_mlx", "ltx_pipelines_mlx", "minimax_h3_mlx", "seedvr2_torch", "moss_image")


@pytest.fixture(scope="module")
def built_distributions(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Build both distribution formats once for the artifact delivery contract."""
    output_dir = tmp_path_factory.mktemp("zvision-distributions")
    subprocess.run(
        ["uv", "build", "--wheel", "--sdist", "--out-dir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
    )

    wheels = list(output_dir.glob("*.whl"))
    sdists = list(output_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels}"
    assert len(sdists) == 1, f"expected one sdist, found {sdists}"
    return wheels[0], sdists[0]


def _license_headers(metadata: bytes) -> list[str]:
    return BytesParser().parsebytes(metadata).get_all("License-File", [])


def _wheel_dist_info(archive: zipfile.ZipFile) -> str:
    metadata_members = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    assert len(metadata_members) == 1, f"expected one dist-info metadata member, found {metadata_members}"
    return metadata_members[0].removesuffix("/METADATA")


def _sdist_root(archive: tarfile.TarFile) -> str:
    roots = {member.name.split("/", 1)[0] for member in archive.getmembers() if member.name}
    assert len(roots) == 1, f"expected one versioned sdist root, found {roots}"
    return roots.pop()


def _read_sdist_member(archive: tarfile.TarFile, member_name: str) -> bytes:
    member = archive.extractfile(member_name)
    assert member is not None, f"missing sdist member {member_name}"
    return member.read()


def test_distributions_include_exact_license_notices_and_metadata(built_distributions: tuple[Path, Path]) -> None:
    wheel_path, sdist_path = built_distributions
    expected_documents = {name: (REPO_ROOT / name).read_bytes() for name in LICENSE_FILES}

    with zipfile.ZipFile(wheel_path) as wheel:
        dist_info = _wheel_dist_info(wheel)
        license_members = {member.removeprefix(f"{dist_info}/licenses/") for member in wheel.namelist() if member.startswith(f"{dist_info}/licenses/")}
        assert license_members == set(LICENSE_FILES)
        for name, expected_bytes in expected_documents.items():
            assert wheel.read(f"{dist_info}/licenses/{name}") == expected_bytes

        metadata = BytesParser().parsebytes(wheel.read(f"{dist_info}/METADATA"))
        assert metadata["License-Expression"] == LICENSE_EXPRESSION
        assert sorted(_license_headers(wheel.read(f"{dist_info}/METADATA"))) == sorted(LICENSE_FILES)

        wheel_members = set(wheel.namelist())
        for vendored in VENDORED_PACKAGES:
            assert any(member.startswith(f"{vendored}/") for member in wheel_members), vendored

    with tarfile.open(sdist_path, "r:gz") as sdist:
        root = _sdist_root(sdist)
        assert {f"{root}/{name}" for name in LICENSE_FILES}.issubset(member.name for member in sdist.getmembers())
        for name, expected_bytes in expected_documents.items():
            assert _read_sdist_member(sdist, f"{root}/{name}") == expected_bytes

        pkg_info = _read_sdist_member(sdist, f"{root}/PKG-INFO")
        metadata = BytesParser().parsebytes(pkg_info)
        assert metadata["License-Expression"] == LICENSE_EXPRESSION
        assert sorted(_license_headers(pkg_info)) == sorted(LICENSE_FILES)


def test_workflow_contract_exposes_canonical_values_only() -> None:
    contract = build_workflow_contract()

    assert contract["values"] == ["txt2img", "img2img", "txt2vid", "img2vid", "extend"]
    assert list(contract["definitions"]) == contract["values"]


def test_extend_definition_lists_its_controls_and_cleared_fields() -> None:
    definition = build_workflow_contract()["definitions"]["extend"]

    assert (definition["mode"], definition["model_kind"]) == ("video", "video")
    assert definition["visible_controls"] == [
        "workflow",
        "model",
        "loras",
        "prompt_source",
        "prompt_scene",
        "prompt_inline",
        "prompt_enhance",
        "prompt_enhance_auto",
        "prompt_file_path",
        "prompt_file_option",
        "prompt_file_preview",
        "prompt_file_edit",
        "source_video",
        "source_video_path",
        "source_video_clear",
        "extend_length",
        "runs",
        "steps",
        "seed",
        "audio",
        "low_memory",
    ]
    assert (definition["supports_reference_image"], definition["requires_reference_image"], definition["requires_source_video"]) == (False, False, True)
    assert set(definition["clear_fields"]) >= {"ratio", "size", "width", "height", "frames", "image_path", "upscale", "video_upscale_enabled", "video_upscale_factor", "negative_prompt", "guidance"}


def test_extend_definition_drops_audio_when_video_has_no_audio_track() -> None:
    without_audio = build_workflow_contract(video_audio=False)["definitions"]["extend"]

    assert "audio" not in without_audio["visible_controls"]
    assert "source_video" in without_audio["visible_controls"]


def test_only_extend_requires_a_source_video() -> None:
    definitions = build_workflow_contract()["definitions"]

    assert {name for name, definition in definitions.items() if definition["requires_source_video"]} == {"extend"}
    assert all(not any(control.startswith("source_video") for control in definitions[name]["visible_controls"]) for name in ("txt2img", "img2img", "txt2vid", "img2vid"))


def test_workflow_contract_hides_audio_when_video_has_no_audio_track() -> None:
    with_audio = build_workflow_contract()["definitions"]
    without_audio = build_workflow_contract(video_audio=False)["definitions"]

    for workflow in ("txt2vid", "img2vid"):
        assert "audio" in with_audio[workflow]["visible_controls"]
        assert "audio" not in without_audio[workflow]["visible_controls"]
        assert "low_memory" in without_audio[workflow]["visible_controls"]


def test_workflow_contract_offers_the_scene_control_for_video_workflows_only() -> None:
    definitions = build_workflow_contract()["definitions"]

    for workflow in ("txt2vid", "img2vid"):
        controls = definitions[workflow]["visible_controls"]
        assert "prompt_scene" in controls
        assert controls.index("prompt_scene") < controls.index("prompt_inline")
    for workflow in ("txt2img", "img2img"):
        assert "prompt_scene" not in definitions[workflow]["visible_controls"]


def test_workflow_contract_accepts_only_canonical_values() -> None:
    assert canonicalize_workflow("txt2img") == "txt2img"
    assert canonicalize_workflow("img2img") == "img2img"
    assert canonicalize_workflow("txt2vid") == "txt2vid"
    assert canonicalize_workflow("img2vid") == "img2vid"
    assert canonicalize_workflow("extend") == "extend"
    assert workflow_mode("extend") == "video"
    assert canonicalize_workflow("image") is None
    assert canonicalize_workflow("texttoimage") is None
    assert canonicalize_workflow("i2i") is None
    assert canonicalize_workflow("i2v") is None
