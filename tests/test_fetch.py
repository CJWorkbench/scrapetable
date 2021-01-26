import tempfile
from pathlib import Path

from cjwmodule.http import httpfile
from cjwmodule.testing.i18n import cjwmodule_i18n_message
from pytest_httpx import HTTPXMock

from scrapetable import FetchResult, fetch_arrow


def P(url="", tablenum=1, first_row_is_header=False):
    return dict(url=url, tablenum=tablenum, first_row_is_header=first_row_is_header)


def test_fetch_happy_path(httpx_mock: HTTPXMock):
    body = b"<html><body><table><thead><tr><th>A</th></tr></thead><tbody><tr><td>a</td></tr></tbody></body></html>"
    httpx_mock.add_response(url="http://example.org", data=body)

    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        result = fetch_arrow(P(url="http://example.org"), {}, None, None, path)
        assert tuple(result) == FetchResult(path, [])
        with httpfile.read(result.path) as (_, __, ___, body_path):
            assert body_path.read_bytes() == body


def test_fetch_http_error(httpx_mock: HTTPXMock):
    # httpx_mock's default behavior is to raise `httpx.TimeoutException`.
    # ref: https://pypi.org/project/pytest-httpx/#raising-exceptions
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        result = fetch_arrow(
            P(url="http://nope.this.will/not/work"), {}, None, None, path
        )
        assert [e.message for e in result.errors] == [
            cjwmodule_i18n_message("http.errors.HttpErrorTimeout")
        ]
        assert result.path.read_bytes() == b""


def test_fetch_no_url():
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        result = fetch_arrow(P(url=""), {}, None, None, path)
        assert tuple(result) == FetchResult(path, [])
        assert result.path.read_bytes() == b""
