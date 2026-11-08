"""Tests for listing and creating projects under the output folder."""

from __future__ import annotations

import pytest

from zvisiongenerator.web.projects import Project, asset_project, create_project, folder_project, list_projects, project_dir


class TestListProjects:
    def test_lists_only_real_visible_folders_sorted_without_case(self, tmp_path):
        (tmp_path / "beta").mkdir()
        (tmp_path / "Alpha").mkdir()
        (tmp_path / ".web_uploads").mkdir()
        (tmp_path / "note.txt").write_text("x")
        (tmp_path / "linked").symlink_to(tmp_path / "Alpha", target_is_directory=True)

        assert list_projects(tmp_path) == (Project("Alpha", str(tmp_path / "Alpha")), Project("beta", str(tmp_path / "beta")))


class TestCreateProject:
    def test_creates_the_folder_with_the_trimmed_name(self, tmp_path):
        project = create_project(tmp_path, "  Reel  ")

        assert project == Project("Reel", str(tmp_path / "Reel"))
        assert (tmp_path / "Reel").is_dir()

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "   ",
            "a/b",
            "a\\b",
            "..",
            ".hidden",
            "a<b",
            "a>b",
            "a:b",
            'a"b',
            "a|b",
            "a?b",
            "a*b",
            "a\x00b",
            "a\tb",
            "a\x1fb",
            "name.",
            "CON",
            "con",
            "Nul.txt",
            "aux.tar.gz",
            "COM1",
            "com9",
            "LPT1",
            "lpt9.log",
        ],
    )
    def test_refuses_a_bad_name(self, tmp_path, name):
        with pytest.raises(ValueError):
            create_project(tmp_path, name)

        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("name", ["bad/name", "trailing.", ".leading"])
    def test_the_refusal_explains_itself(self, tmp_path, name):
        with pytest.raises(ValueError) as refused:
            create_project(tmp_path, name)

        assert str(refused.value)
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("name", ["Reel", "my project", "a.b", "a..b", "CONSOLE", "COM10", "LPT0", "naïve"])
    def test_accepts_an_ordinary_name(self, tmp_path, name):
        assert create_project(tmp_path, name).name == name

    def test_a_duplicate_ignoring_case_is_refused(self, tmp_path):
        (tmp_path / "reel").mkdir()

        with pytest.raises(FileExistsError, match="Reel"):
            create_project(tmp_path, "Reel")


class TestProjectDir:
    @pytest.mark.parametrize("name", [None, "", "  "])
    def test_a_blank_name_is_the_root(self, tmp_path, name):
        assert project_dir(tmp_path, name) == tmp_path

    def test_a_name_is_a_folder_under_the_root(self, tmp_path):
        assert project_dir(tmp_path, "Reel") == tmp_path / "Reel"

    def test_an_existing_odd_folder_name_is_accepted(self, tmp_path):
        assert project_dir(tmp_path, "a:b") == tmp_path / "a:b"

    @pytest.mark.parametrize("name", ["../x", "..", ".hidden", "a/b", "a\\b", "a\0b"])
    def test_refuses_a_name_that_is_not_one_plain_folder(self, tmp_path, name):
        with pytest.raises(ValueError):
            project_dir(tmp_path, name)


class TestAssetProject:
    def test_the_first_segment_is_the_project(self):
        assert asset_project("A/x.png") == "A"
        assert asset_project("A/sub/x.png") == "A"

    @pytest.mark.parametrize("asset_id", ["x.png", ".web_uploads/x.png", ".hidden/sub/x.png", "/x.png"])
    def test_no_project_for_a_root_or_dot_folder_asset(self, asset_id):
        assert asset_project(asset_id) is None


class TestFolderProject:
    @pytest.mark.parametrize(
        ("folder", "expected"),
        [("", None), ("A", "A"), ("A/sub", "A"), (".web_uploads", None)],
    )
    def test_names_the_project_a_save_into_the_folder_lands_in(self, tmp_path, folder, expected):
        root = tmp_path / "out"
        (root / folder).mkdir(parents=True)

        assert folder_project(root, root / folder) == expected

    def test_a_folder_outside_the_root_is_in_no_project(self, tmp_path):
        (tmp_path / "out").mkdir()

        assert folder_project(tmp_path / "out", tmp_path) is None
