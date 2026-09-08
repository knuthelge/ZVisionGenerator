from __future__ import annotations

from email.parser import BytesParser
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

import pytest

from zvisiongenerator.web.workspace_contract import build_workflow_contract, canonicalize_workflow


REPO_ROOT = Path(__file__).resolve().parents[1]
LICENSE_FILES = ("LICENSE", "THIRD_PARTY_LICENSES.md")
LICENSE_EXPRESSION = "AGPL-3.0-or-later"


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
        assert any(member.startswith("ltx_core_mlx/") for member in wheel_members)
        assert any(member.startswith("ltx_pipelines_mlx/") for member in wheel_members)

    with tarfile.open(sdist_path, "r:gz") as sdist:
        root = _sdist_root(sdist)
        assert {f"{root}/{name}" for name in LICENSE_FILES}.issubset(member.name for member in sdist.getmembers())
        for name, expected_bytes in expected_documents.items():
            assert _read_sdist_member(sdist, f"{root}/{name}") == expected_bytes

        pkg_info = _read_sdist_member(sdist, f"{root}/PKG-INFO")
        metadata = BytesParser().parsebytes(pkg_info)
        assert metadata["License-Expression"] == LICENSE_EXPRESSION
        assert sorted(_license_headers(pkg_info)) == sorted(LICENSE_FILES)


def test_make_check_enforces_frontend_docs_and_packaged_spa_gates() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    match = re.search(r"^check:\s+(?P<deps>.+?)\s+##", makefile, flags=re.MULTILINE)

    assert match is not None
    deps = match.group("deps").split()

    assert "frontend-test" in deps
    assert "frontend-static-check" in deps
    assert "docs-check" in deps


def test_make_install_installs_python_and_frontend_dependencies() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    install_match = re.search(r"^install:\s+(?P<deps>[^#\n]+)##[^\n]*(?P<body>(?:\n\t.+)+)", makefile, flags=re.MULTILINE)
    frontend_match = re.search(r"^frontend-install:[^#\n]*##[^\n]*(?P<body>(?:\n\t.+)+)", makefile, flags=re.MULTILINE)

    assert install_match is not None
    assert frontend_match is not None
    assert "frontend-install" in install_match.group("deps").split()
    assert "uv sync" in install_match.group("body")
    assert "pnpm --dir frontend install --frozen-lockfile" in frontend_match.group("body")


def test_release_workflow_validates_packaged_spa_before_build() -> None:
    release_workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    validation_index = release_workflow.index("make frontend-static-check")
    build_index = release_workflow.index("make build")

    assert validation_index < build_index


def test_workflow_contract_exposes_canonical_values_only() -> None:
    contract = build_workflow_contract()

    assert "legacy_aliases" not in contract
    assert contract["values"] == ["txt2img", "img2img", "txt2vid", "img2vid"]


def test_workflow_contract_accepts_only_canonical_values() -> None:
    assert canonicalize_workflow("txt2img") == "txt2img"
    assert canonicalize_workflow("img2img") == "img2img"
    assert canonicalize_workflow("txt2vid") == "txt2vid"
    assert canonicalize_workflow("img2vid") == "img2vid"
    assert canonicalize_workflow("image") is None
    assert canonicalize_workflow("texttoimage") is None
    assert canonicalize_workflow("i2i") is None
    assert canonicalize_workflow("i2v") is None
