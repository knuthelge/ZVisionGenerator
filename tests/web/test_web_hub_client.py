"""Tests for the anonymous, bounded Hugging Face reader behind Discover LoRAs."""

from __future__ import annotations

from huggingface_hub.utils import httpx
import pytest

from zvisiongenerator.web.hub_client import HEADER_READ_BYTES, CachedFetch, HubError, HubResponse, api_url, http_fetch, resolve_url, validated_file, validated_repo, validated_revision

_COMMIT = "a" * 40


class TestValidation:
    @pytest.mark.parametrize("repo", ["owner/name", "Owner-1/Name_2.v3", "a/b"])
    def test_accepts_repository_ids(self, repo):
        assert validated_repo(repo) == repo

    @pytest.mark.parametrize(
        "repo", ["../x", "owner/../x", "owner/name/extra", "owner", "", "/owner/name", "owner/name?x=1", "owner/na me", "https://huggingface.co/a/b", "owner\\name", "-owner/name"]
    )
    def test_refuses_anything_that_is_not_a_repository_id(self, repo):
        with pytest.raises(ValueError):
            validated_repo(repo)

    @pytest.mark.parametrize("file", ["a.safetensors", "dir/a.safetensors", "[Z].Turbo v1.safetensors"])
    def test_accepts_safetensors_paths(self, file):
        assert validated_file(file) == file

    @pytest.mark.parametrize(
        "file",
        [
            "../a.safetensors",
            "dir/../a.safetensors",
            "/a.safetensors",
            "\\a.safetensors",
            "dir\\a.safetensors",
            "a.bin",
            "a.safetensors/",
            "dir//a.safetensors",
            "a\x00.safetensors",
            "",
            "x" * 400 + ".safetensors",
        ],
    )
    def test_refuses_other_files(self, file):
        with pytest.raises(ValueError):
            validated_file(file)

    def test_only_a_full_commit_hash_is_a_revision(self):
        assert validated_revision(_COMMIT) == _COMMIT
        for bad in ("main", "abc", "A" * 40, _COMMIT + "0"):
            with pytest.raises(ValueError):
                validated_revision(bad)


class TestUrls:
    def test_urls_are_built_on_the_hub_and_encode_the_file(self):
        assert resolve_url("o/r", _COMMIT, "dir/a b#1.safetensors") == f"https://huggingface.co/o/r/resolve/{_COMMIT}/dir/a%20b%231.safetensors"
        assert (
            api_url("models", {"filter": "x:y", "limit": 3, "search": None}, repeated={"expand[]": ("a", "b")})
            == "https://huggingface.co/api/models?filter=x%3Ay&limit=3&expand%5B%5D=a&expand%5B%5D=b"
        )

    def test_resolve_url_validates_every_part(self):
        for args in (("../x", _COMMIT, "a.safetensors"), ("o/r", "main2", "a.safetensors"), ("o/r", _COMMIT, "../a.safetensors")):
            with pytest.raises(ValueError):
                resolve_url(*args)


def _transport(handler):
    return httpx.MockTransport(handler)


class TestHttpFetch:
    def test_sends_no_credential_even_when_a_token_is_in_the_environment(self, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "hf_secret")
        seen = []

        def handler(request):
            seen.append(request)
            return httpx.Response(200, content=b"{}")

        http_fetch("https://huggingface.co/api/models", headers={"Authorization": "Bearer hf_secret", "Cookie": "x=1", "Range": "bytes=0-9"}, transport=_transport(handler))

        assert "authorization" not in seen[0].headers
        assert "cookie" not in seen[0].headers
        assert seen[0].headers["range"] == "bytes=0-9"
        assert "hf_secret" not in str(seen[0].headers) + str(seen[0].url)

    def test_stops_reading_at_the_cap_even_when_the_server_ignores_the_range(self):
        pulled = []

        def body():
            for _ in range(10):
                pulled.append(1)
                yield b"x" * (1024 * 1024)

        response = http_fetch("https://huggingface.co/o/r/resolve/main/a.safetensors", max_bytes=HEADER_READ_BYTES, transport=_transport(lambda request: httpx.Response(200, content=body())))

        assert len(response.body) == HEADER_READ_BYTES == 2 * 1024 * 1024
        assert response.truncated is True
        assert len(pulled) <= 3

    def test_a_short_answer_is_complete(self):
        response = http_fetch("https://huggingface.co/api/models", transport=_transport(lambda request: httpx.Response(200, content=b'{"a": 1}', headers={"Link": "<x>"})))

        assert (response.status, response.body, response.truncated, response.headers["link"]) == (200, b'{"a": 1}', False, "<x>")

    @pytest.mark.parametrize(
        "url",
        [
            "http://huggingface.co/api/models",
            "https://example.com/api/models",
            "https://huggingface.co.evil.test/x",
            "https://user@huggingface.co/x",
            "file:///etc/passwd",
            "https://evilhuggingface.co/x",
        ],
    )
    def test_refuses_urls_off_the_hub(self, url):
        def handler(request):
            raise AssertionError("must not be requested")

        with pytest.raises(ValueError):
            http_fetch(url, transport=_transport(handler))

    def test_follows_a_redirect_to_the_cdn_and_keeps_the_range_header(self):
        seen = []

        def handler(request):
            seen.append((str(request.url), request.headers.get("range")))
            if request.url.host == "huggingface.co":
                return httpx.Response(302, headers={"location": "https://us.aws.cdn.hf.co/blob?sig=1"})
            return httpx.Response(206, content=b"abc")

        response = http_fetch("https://huggingface.co/o/r/resolve/main/a.safetensors", headers={"Range": "bytes=0-2"}, transport=_transport(handler))

        assert response.body == b"abc"
        assert seen == [("https://huggingface.co/o/r/resolve/main/a.safetensors", "bytes=0-2"), ("https://us.aws.cdn.hf.co/blob?sig=1", "bytes=0-2")]

    @pytest.mark.parametrize("location", ["https://evil.example/x", "http://huggingface.co/x", "https://huggingface.co.evil.example/x"])
    def test_refuses_a_redirect_off_the_hub(self, location):
        with pytest.raises(ValueError):
            http_fetch("https://huggingface.co/x", transport=_transport(lambda request: httpx.Response(302, headers={"location": location})))

    def test_a_redirect_loop_ends_in_an_error(self):
        with pytest.raises(HubError):
            http_fetch("https://huggingface.co/x", transport=_transport(lambda request: httpx.Response(302, headers={"location": "https://huggingface.co/x"})))

    def test_a_connection_failure_is_a_hub_error_without_a_status(self):
        def handler(request):
            raise httpx.ConnectError("down")

        with pytest.raises(HubError) as raised:
            http_fetch("https://huggingface.co/x", transport=_transport(handler))

        assert raised.value.status is None


class TestCachedFetch:
    def test_a_repeated_read_is_served_from_memory_until_it_expires(self):
        now = [0.0]
        calls = []

        def fetch(url, *, headers=None, max_bytes=0):
            calls.append(url)
            return HubResponse(200, b"x", {})

        cached = CachedFetch(fetch, ttl_secs=60, clock=lambda: now[0])
        cached("https://huggingface.co/a")
        cached("https://huggingface.co/a")
        cached("https://huggingface.co/a", headers={"Range": "bytes=0-1"})
        now[0] = 61
        cached("https://huggingface.co/a")

        assert calls == ["https://huggingface.co/a", "https://huggingface.co/a", "https://huggingface.co/a"]

    def test_an_error_answer_is_not_kept(self):
        calls = []

        def fetch(url, *, headers=None, max_bytes=0):
            calls.append(url)
            return HubResponse(500, b"", {})

        cached = CachedFetch(fetch)
        cached("https://huggingface.co/a")
        cached("https://huggingface.co/a")

        assert len(calls) == 2

    def test_the_least_recently_used_entry_goes_first(self):
        calls = []

        def fetch(url, *, headers=None, max_bytes=0):
            calls.append(url)
            return HubResponse(200, b"", {})

        cached = CachedFetch(fetch, max_entries=2)
        for url in ("https://huggingface.co/a", "https://huggingface.co/b", "https://huggingface.co/a", "https://huggingface.co/c", "https://huggingface.co/a", "https://huggingface.co/b"):
            cached(url)

        assert calls == ["https://huggingface.co/a", "https://huggingface.co/b", "https://huggingface.co/c", "https://huggingface.co/b"]

    def test_a_body_too_large_to_be_worth_keeping_is_read_again(self):
        calls = []

        def fetch(url, *, headers=None, max_bytes=0):
            calls.append(url)
            return HubResponse(206, b"x" * 101, {})

        cached = CachedFetch(fetch, max_entry_bytes=100)
        cached("https://huggingface.co/a")
        cached("https://huggingface.co/a")

        assert len(calls) == 2

    def test_the_bodies_kept_never_add_up_to_more_than_the_byte_cap(self):
        calls = []

        def fetch(url, *, headers=None, max_bytes=0):
            calls.append(url)
            return HubResponse(200, b"x" * 40, {})

        cached = CachedFetch(fetch, max_total_bytes=100)
        for url in ("https://huggingface.co/a", "https://huggingface.co/b", "https://huggingface.co/c"):
            cached(url)
        cached("https://huggingface.co/c")
        cached("https://huggingface.co/b")
        cached("https://huggingface.co/a")

        # Reading c pushed a out (120 > 100 bytes); c and b were still kept, so only the second read of a goes to the Hub.
        assert calls == ["https://huggingface.co/a", "https://huggingface.co/b", "https://huggingface.co/c", "https://huggingface.co/a"]

    def test_expired_answers_are_dropped_from_memory_even_if_never_asked_for_again(self):
        now = [0.0]

        def fetch(url, *, headers=None, max_bytes=0):
            return HubResponse(200, b"x" * 10, {})

        cached = CachedFetch(fetch, ttl_secs=60, clock=lambda: now[0])
        cached("https://huggingface.co/a")
        cached("https://huggingface.co/b")
        now[0] = 61
        cached("https://huggingface.co/c")

        assert cached._bytes == 10
        assert len(cached._entries) == 1
