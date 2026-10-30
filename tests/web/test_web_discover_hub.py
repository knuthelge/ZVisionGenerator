"""Tests for reading LoRA listings, repository details and file families from a (fake) Hugging Face."""

from __future__ import annotations

from functools import partial
from urllib.parse import parse_qs, urlsplit

import pytest

from tests.web._helpers import COMMIT, SHA, FakeHub, model_json, safetensors_bytes
from zvisiongenerator.web import discover_hub
from zvisiongenerator.web.discover_hub import BASE_MODELS, RepoFile, base_repos, check_family, choose_files, clean_repo_name, preview_url
from zvisiongenerator.web.hub_client import HubError, HubResponse
from zvisiongenerator.web.listing_filter import ListingFilter

_ZIMAGE = "Tongyi-MAI/Z-Image-Turbo"
# Made-up terms: a whole word and a stem.
_RULES = ListingFilter(frozenset({"zap"}), ("zzpix",))
list_adapters = partial(discover_hub.list_adapters, rules=_RULES)
read_repo = partial(discover_hub.read_repo, rules=_RULES)


def _item(repo: str, **fields) -> dict:
    data = {
        "id": repo,
        "downloads": 10,
        "likes": 1,
        "gated": False,
        "tags": ["lora"],
        "siblings": [{"rfilename": "a.safetensors"}],
        "cardData": {"license": "mit"},
        "createdAt": "2026-01-01T00:00:00.000Z",
    }
    data.update(fields)
    return data


def _file(name: str, *, step: int | None = None, marked: bool = False) -> RepoFile:
    return RepoFile(name, 10, None, step, marked)


class TestListing:
    def test_asks_the_hub_for_adapters_of_the_official_base_model_with_the_chosen_sort_and_search(self):
        hub = FakeHub(listing={_ZIMAGE: ([_item("o/a")], None)})

        page = list_adapters(hub, family="zimage", query=" pixel ", sort="likes")

        query = parse_qs(urlsplit(hub.urls[0]).query)
        assert query["filter"] == [f"base_model:adapter:{_ZIMAGE}"]
        assert (query["sort"], query["direction"], query["search"]) == (["likes"], ["-1"], ["pixel"])
        assert {"gated", "cardData", "siblings"} <= set(query["expand[]"])
        assert [row["repo"] for row in page["items"]] == ["o/a"]
        assert page["base_models"] == [_ZIMAGE]

    def test_gated_private_adult_and_fileless_repositories_are_never_listed(self):
        items = [
            _item("o/ok"),
            _item("o/gated", gated="auto"),
            _item("o/manual", gated="manual"),
            _item("o/private", private=True),
            _item("o/adult", tags=["lora", "not-for-all-audiences"]),
            _item("o/nofiles", siblings=[{"rfilename": "README.md"}]),
            _item("../x"),
        ]

        page = list_adapters(FakeHub(listing={_ZIMAGE: (items, None)}), family="zimage")

        assert [row["repo"] for row in page["items"]] == ["o/ok"]

    @pytest.mark.parametrize(
        "fields",
        [
            {"id": "o/MysticZap"},
            {"id": "o/ZAP_v2"},
            {"id": "o/krea2zzpixmix"},
            {"tags": ["lora", "zap"]},
            {"cardData": {"tags": ["Zzpix"]}},
            {"cardData": {"instance_prompt": ["zap", "studio"]}},
            {"cardData": {"instance_prompt": "zzpix style"}},
            {"siblings": [{"rfilename": "style.safetensors"}, {"rfilename": "images/zzpix_01.png"}]},
            {"cardData": {"tags": ["not-for-all-audiences"]}},
        ],
    )
    def test_a_repository_the_listing_filter_keeps_out_is_never_listed(self, fields):
        page = list_adapters(FakeHub(listing={_ZIMAGE: ([_item("o/r", **fields)], None)}), family="zimage")

        assert page["items"] == []

    def test_an_adult_tagged_repository_is_listed_when_the_rules_do_not_hide_the_tag(self):
        item = _item("o/r", tags=["lora", "not-for-all-audiences"])

        page = discover_hub.list_adapters(FakeHub(listing={_ZIMAGE: ([item], None)}), family="zimage", rules=ListingFilter(frozenset(), (), hide_adult_tagged=False))

        assert [row["repo"] for row in page["items"]] == ["o/r"]

    def test_a_word_inside_another_word_does_not_keep_a_repository_out(self):
        page = list_adapters(FakeHub(listing={_ZIMAGE: ([_item("o/zapper-style")], None)}), family="zimage")

        assert [row["repo"] for row in page["items"]] == ["o/zapper-style"]

    def test_a_page_with_every_row_left_out_reads_the_next_hub_page(self):
        hub = FakeHub(pages={(_ZIMAGE, ""): ([_item("o/zap")], "C1"), (_ZIMAGE, "C1"): ([_item("o/ok")], "C2")})

        page = list_adapters(hub, family="zimage")

        assert [row["repo"] for row in page["items"]] == ["o/ok"]
        assert page["next_cursor"] is not None
        assert len(hub.calls) == 2

    def test_reading_more_hub_pages_stops_after_the_extra_pages(self):
        hub = FakeHub(listing={_ZIMAGE: ([_item("o/zap")], "C1")})

        page = list_adapters(hub, family="zimage")

        assert page["items"] == []
        assert page["next_cursor"] is not None
        assert len(hub.calls) == 1 + discover_hub._EXTRA_PAGES

    def test_a_row_carries_what_the_list_shows(self):
        card = {"license": "apache-2.0", "widget": [{"output": {"url": "images/p.png"}}, {"output": {"url": "https://evil.example/x.png"}}]}
        item = _item("renderartist/Technically-Color-Z", cardData=card, siblings=[{"rfilename": "a.safetensors"}, {"rfilename": "b.safetensors"}], downloads=7, likes=3)

        row = list_adapters(FakeHub(listing={_ZIMAGE: ([item], None)}), family="zimage")["items"][0]

        assert row["name"] == "Technically Color Z"
        assert (row["author"], row["downloads"], row["likes"], row["files"], row["licence"]) == ("renderartist", 7, 3, 2, "apache-2.0")
        assert row["thumbnail"] == "https://huggingface.co/renderartist/Technically-Color-Z/resolve/main/images/p.png"

    def test_a_cursor_leads_to_the_next_page_and_the_last_page_has_none(self):
        hub = FakeHub(listing={_ZIMAGE: ([_item("o/a")], "HUBCURSOR==")})

        first = list_adapters(hub, family="zimage")
        hub.listing[_ZIMAGE] = ([_item("o/b")], None)
        second = list_adapters(hub, family="zimage", cursor=first["next_cursor"])

        assert first["next_cursor"] is not None
        assert parse_qs(urlsplit(hub.urls[1]).query)["cursor"] == ["HUBCURSOR=="]
        assert [row["repo"] for row in second["items"]] == ["o/b"]
        assert second["next_cursor"] is None

    @pytest.mark.parametrize("cursor", ["!!!", "bm90IGpzb24=", "W1siZXZpbC9iYXNlIiwgIngiXV0="])
    def test_a_cursor_not_made_by_the_server_is_refused_before_the_hub_is_asked(self, cursor):
        hub = FakeHub()

        with pytest.raises(ValueError, match="cursor"):
            list_adapters(hub, family="zimage", cursor=cursor)

        assert hub.calls == []

    def test_ltx_lists_both_versions_together_in_download_order(self):
        hub = FakeHub(listing={"Lightricks/LTX-2.3": ([_item("o/small", downloads=5)], None), "Lightricks/LTX-2.5": ([_item("o/big", downloads=50)], None)})

        page = list_adapters(hub, family="ltx")

        assert [row["repo"] for row in page["items"]] == ["o/big", "o/small"]
        assert page["base_models"] == ["Lightricks/LTX-2.3", "Lightricks/LTX-2.5"]

    @pytest.mark.parametrize(("version", "base", "other"), [("2.3", "Lightricks/LTX-2.3", "Lightricks/LTX-2.5"), ("2.5", "Lightricks/LTX-2.5", "Lightricks/LTX-2.3")])
    def test_ltx_lists_only_the_adapters_of_the_chosen_version(self, version, base, other):
        hub = FakeHub(listing={base: ([_item("o/mine")], None), other: ([_item("o/theirs")], None)})

        page = list_adapters(hub, family="ltx", version=version)

        assert [row["repo"] for row in page["items"]] == ["o/mine"]
        assert page["base_models"] == [base]
        assert [parse_qs(urlsplit(url).query)["filter"] for url in hub.urls] == [[f"base_model:adapter:{base}"]]

    def test_an_unknown_ltx_version_is_refused(self):
        with pytest.raises(ValueError):
            list_adapters(FakeHub(), family="ltx", version="9.9")

    def test_a_repository_that_is_an_adapter_of_several_bases_is_listed_once(self):
        shared = _item("o/shared", downloads=40)
        hub = FakeHub(listing={"Lightricks/LTX-2.3": ([shared, _item("o/small", downloads=5)], None), "Lightricks/LTX-2.5": ([shared, _item("o/big", downloads=50)], None)})

        page = list_adapters(hub, family="ltx")

        assert [row["repo"] for row in page["items"]] == ["o/big", "o/shared", "o/small"]

    def test_klein_lists_the_repository_of_the_chosen_size(self):
        assert base_repos("flux2_klein", "9b") == ("black-forest-labs/FLUX.2-klein-9B",)
        assert len(base_repos("flux2_klein")) == 2

    @pytest.mark.parametrize(("family", "size", "sort"), [("flux9", None, "downloads"), ("flux2_klein", "70b", "downloads"), ("zimage", None, "random")])
    def test_an_unknown_family_size_or_sort_is_refused(self, family, size, sort):
        with pytest.raises(ValueError):
            list_adapters(FakeHub(), family=family, size=size, sort=sort)

    def test_a_hub_error_is_reported(self):
        def failing(url, **_):
            return HubResponse(500, b"", {})

        with pytest.raises(HubError) as raised:
            list_adapters(failing, family="zimage")

        assert raised.value.status == 500

    def test_every_curated_family_has_a_base_model_to_search(self):
        from zvisiongenerator.utils.curated_loras import CURATED_FAMILIES

        assert set(CURATED_FAMILIES) <= set(BASE_MODELS)


class TestPreviewUrls:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("images/a b.png", "https://huggingface.co/o/r/resolve/main/images/a%20b.png"),
            ("/images/a.webp", "https://huggingface.co/o/r/resolve/main/images/a.webp"),
            ("https://huggingface.co/o/r/resolve/main/x.jpg?download=true", "https://huggingface.co/o/r/resolve/main/x.jpg"),
            ("https://huggingface.co/o/r/blob/main/x.jpg", "https://huggingface.co/o/r/resolve/main/x.jpg"),
            ("https://cdn-uploads.huggingface.co/production/uploads/abc/x.png", "https://cdn-uploads.huggingface.co/production/uploads/abc/x.png"),
            ("videos/clip.mp4", "https://huggingface.co/o/r/resolve/main/videos/clip.mp4"),
        ],
    )
    def test_a_preview_on_the_hub_becomes_a_resolve_url(self, raw, expected):
        assert preview_url(raw, "o/r") == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "https://evil.example/x.png",
            "http://huggingface.co/o/r/resolve/main/x.png",
            "https://huggingface.co.evil.example/x.png",
            "../x.png",
            "images/../../x.png",
            "data:image/png;base64,AAAA",
            "javascript:alert(1)",
            "images/a.exe",
            "images/",
            None,
            3,
            "",
        ],
    )
    def test_a_preview_elsewhere_or_of_another_kind_is_dropped(self, raw):
        assert preview_url(raw, "o/r") is None


class TestChooseFiles:
    def test_a_marked_file_is_preselected_and_the_other_snapshots_fold(self):
        files = [_file("s_000001500.safetensors", step=1500), _file("s_000002500_(recommended).safetensors", step=2500, marked=True), _file("s_000003500.safetensors", step=3500)]

        chosen, choices, folded = choose_files(files)

        assert chosen is files[1]
        assert choices == (files[1],)
        assert folded == (files[0], files[2])

    def test_without_a_marker_the_file_without_a_step_is_preselected_and_real_variants_stay_choices(self):
        files = [_file("v_000001000.safetensors", step=1000), _file("dejpeg_a.safetensors"), _file("dejpeg_b.safetensors")]

        chosen, choices, folded = choose_files(files)

        assert chosen is files[1]
        assert choices == (files[1], files[2])
        assert folded == (files[0],)

    def test_with_only_snapshots_the_highest_step_is_preselected(self):
        files = [_file("a_000001000.safetensors", step=1000), _file("a_000003000.safetensors", step=3000), _file("a_000002000.safetensors", step=2000)]

        chosen, choices, folded = choose_files(files)

        assert chosen is files[1]
        assert choices == (files[1],)
        assert len(folded) == 2

    def test_no_files_choose_nothing(self):
        assert choose_files([]) == (None, (), ())


class TestReadRepo:
    def test_reads_the_commit_card_previews_and_files(self):
        card = {
            "license": "apache-2.0",
            "base_model": ["Tongyi-MAI/Z-Image-Turbo"],
            "instance_prompt": "t3chnic4lly, 1960s film",
            "widget": [{"output": {"url": "images/a.png"}}, {"output": {"url": "images/a.png"}}, {"output": {"url": "https://evil.example/b.png"}}, {"output": {"url": "v/c.mp4"}}],
        }
        files = {"my_final.safetensors": 2000, "step_000001000.safetensors": 3000, "notes.txt": 1}
        hub = FakeHub(models={"o/r": model_json("o/r", cardData=card, files=files)})

        detail = read_repo(hub, "o/r")

        assert (detail.revision, detail.licence, detail.base_model, detail.instance_prompt) == (COMMIT, "apache-2.0", "Tongyi-MAI/Z-Image-Turbo", "t3chnic4lly, 1960s film")
        assert detail.previews == (f"https://huggingface.co/o/r/resolve/{COMMIT}/images/a.png",)
        assert detail.preview_videos == (f"https://huggingface.co/o/r/resolve/{COMMIT}/v/c.mp4",)
        assert [(entry.file, entry.size_bytes, entry.sha256, entry.step, entry.marked) for entry in detail.files] == [
            ("my_final.safetensors", 2000, SHA, None, True),
            ("step_000001000.safetensors", 3000, SHA, 1000, False),
        ]

    @pytest.mark.parametrize("fields", [{"gated": "auto"}, {"gated": "manual"}, {"tags": ["lora", "not-for-all-audiences"]}, {"private": True}])
    def test_a_gated_private_or_adult_repository_is_refused(self, fields):
        with pytest.raises(ValueError):
            read_repo(FakeHub(models={"o/r": model_json("o/r", **fields)}), "o/r")

    def test_a_repository_the_listing_filter_keeps_out_is_refused(self):
        with pytest.raises(ValueError):
            read_repo(FakeHub(models={"o/r": model_json("o/r", files={"zzpix.safetensors": 1})}), "o/r")

    def test_a_missing_repository_is_a_404(self):
        with pytest.raises(HubError) as raised:
            read_repo(FakeHub(), "o/r")

        assert raised.value.status == 404

    def test_a_bad_repo_id_never_reaches_the_hub(self):
        hub = FakeHub()

        with pytest.raises(ValueError):
            read_repo(hub, "../x")

        assert hub.calls == []

    def test_a_file_name_that_cannot_be_downloaded_is_left_out(self):
        hub = FakeHub(models={"o/r": model_json("o/r", files={"ok.safetensors": 1, "../evil.safetensors": 1, "/abs.safetensors": 1})})

        assert [entry.file for entry in read_repo(hub, "o/r").files] == ["ok.safetensors"]

    def test_a_card_without_a_licence_or_prompt_reads_as_none(self):
        detail = read_repo(FakeHub(models={"o/r": model_json("o/r", cardData={})}), "o/r")

        assert (detail.licence, detail.base_model, detail.instance_prompt) == (None, None, None)


class TestCheckFamily:
    def _hub(self, body: bytes) -> FakeHub:
        return FakeHub(files={("o/r", "a.safetensors"): body})

    def test_a_file_made_for_the_model_matches(self):
        hub = self._hub(safetensors_bytes(metadata={"ss_base_model_version": "zimage"}))

        result = check_family(hub, repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage")

        assert (result.status, result.family) == ("match", "zimage")

    def test_a_file_made_for_another_model_is_a_mismatch_that_names_it(self):
        hub = self._hub(safetensors_bytes(metadata={"ss_base_model_version": "krea2"}))

        result = check_family(hub, repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage")

        assert (result.status, result.family) == ("mismatch", "krea2")

    def test_a_klein_file_of_the_other_size_is_a_mismatch(self):
        hub = self._hub(safetensors_bytes(metadata={"ss_base_model_version": "klein_4b"}))

        assert check_family(hub, repo="o/r", revision=COMMIT, file="a.safetensors", family="flux2_klein", size="9b").status == "mismatch"
        assert check_family(hub, repo="o/r", revision=COMMIT, file="a.safetensors", family="flux2_klein", size="4b").status == "match"

    @pytest.mark.parametrize("body", [safetensors_bytes(), b"not a safetensors file", b"\xff" * 20, safetensors_bytes(metadata={"ss_base_model_version": "something"})])
    def test_a_header_that_names_no_family_or_cannot_be_read_is_unknown(self, body):
        result = check_family(self._hub(body), repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage")

        assert (result.status, result.family) == ("unknown", None)

    def test_reads_the_length_and_then_exactly_the_header_by_range(self):
        body = safetensors_bytes(metadata={"ss_base_model_version": "zimage"}, tail=5 * 1024 * 1024)
        hub = self._hub(body)
        header_bytes = int.from_bytes(body[:8], "little")

        check_family(hub, repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage")

        url = f"https://huggingface.co/o/r/resolve/{COMMIT}/a.safetensors"
        assert hub.calls == [
            (url, {"Range": "bytes=0-7"}, 8),
            (url, {"Range": f"bytes=8-{8 + header_bytes - 1}"}, header_bytes),
        ]

    def test_a_server_that_ignores_the_range_never_has_more_than_the_length_asked_for_read(self):
        class Whole:
            calls = 0

            def __call__(self, url, *, headers=None, max_bytes=0):
                self.calls += 1
                return HubResponse(200, safetensors_bytes(metadata={"ss_base_model_version": "zimage"})[:max_bytes], {})

        # The second read cannot be answered from the start of the file, so the header is not guessed at.
        assert check_family(Whole(), repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage").status == "unknown"

    def test_a_header_longer_than_the_cap_is_unknown_not_an_error(self):
        huge = (10 * 1024 * 1024).to_bytes(8, "little") + b"{" + b" " * 100
        hub = self._hub(huge)

        assert check_family(hub, repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage").status == "unknown"
        assert len(hub.calls) == 1

    @pytest.mark.parametrize(
        ("repo", "revision", "file"),
        [("../x", COMMIT, "a.safetensors"), ("o/r", COMMIT, "../a.safetensors"), ("o/r", "main", "a.safetensors"), ("o/r", COMMIT, "/a.safetensors"), ("o/r", COMMIT, "a.bin")],
    )
    def test_bad_input_never_reaches_the_hub(self, repo, revision, file):
        hub = self._hub(b"")

        with pytest.raises(ValueError):
            check_family(hub, repo=repo, revision=revision, file=file, family="zimage")

        assert hub.calls == []

    def test_a_missing_file_is_a_hub_error(self):
        with pytest.raises(HubError) as raised:
            check_family(FakeHub(), repo="o/r", revision=COMMIT, file="a.safetensors", family="zimage")

        assert raised.value.status == 404


def test_clean_repo_name_reads_as_words():
    assert clean_repo_name("owner/Pixel_Art--Redmond") == "Pixel Art Redmond"
