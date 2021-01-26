import contextlib
import io
import tempfile
from pathlib import Path
from typing import ContextManager, List, NamedTuple, Optional, Tuple

import pyarrow as pa
from cjwmodule.http import httpfile
from cjwmodule.testing.i18n import cjwmodule_i18n_message, i18n_message

from scrapetable import FetchResult, RenderError, render


class DefaultSettings(NamedTuple):
    MAX_ROWS_PER_TABLE: int = 1000
    MAX_COLUMNS_PER_TABLE: int = 10
    MAX_BYTES_PER_VALUE: int = 10000
    MAX_BYTES_TEXT_DATA: int = 100000
    MAX_BYTES_PER_COLUMN_NAME: int = 100
    MAX_CSV_BYTES: int = 1000000
    MAX_DICTIONARY_PYLIST_N_BYTES: int = 1000
    MIN_DICTIONARY_COMPRESSION_RATIO_PYLIST_N_BYTES: float = 2.0


def P(url="", tablenum=1, first_row_is_header=False):
    return dict(url=url, tablenum=tablenum, first_row_is_header=first_row_is_header)


def _assert_table_file(path: Path, expected: Optional[pa.Table]) -> None:
    if expected is None:
        assert path.stat().st_size == 0
        return
    else:
        assert path.stat().st_size > 0

    with pa.ipc.open_file(path) as f:
        actual = f.read_all()
    assert actual.column_names == expected.column_names
    for actual_column, expected_column in zip(
        actual.itercolumns(), expected.itercolumns()
    ):
        assert actual_column.type == expected_column.type
        assert actual_column.to_pylist() == expected_column.to_pylist()
        if pa.types.is_dictionary(actual_column.type):
            for output_chunk, expected_chunk in zip(
                actual_column.iterchunks(), expected_column.iterchunks()
            ):
                assert (
                    output_chunk.dictionary.to_pylist()
                    == expected_chunk.dictionary.to_pylist()
                )


@contextlib.contextmanager
def _temp_parquet_file(table: pa.Table) -> ContextManager[Path]:
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        pa.parquet.write_table(table, path, version="2.0", compression="SNAPPY")
        yield path


@contextlib.contextmanager
def _temp_httpfile(
    url: str,
    status_line: str,
    body: bytes,
    headers: List[Tuple[str, str]] = [("Content-Type", "text/html; charset=utf-8")],
) -> ContextManager[Path]:
    with tempfile.NamedTemporaryFile() as tf:
        path = Path(tf.name)
        httpfile.write(path, {"url": url}, status_line, headers, io.BytesIO(body))
        yield path


def test_render_empty():
    with tempfile.NamedTemporaryFile() as fetch_tf, tempfile.NamedTemporaryFile() as render_tf:
        fetch_result = FetchResult(Path(fetch_tf.name), [])
        render_path = Path(render_tf.name)
        result = render(
            pa.table({}),
            P(),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(render_path, None)


def test_render_error():
    with tempfile.NamedTemporaryFile() as fetch_tf, tempfile.NamedTemporaryFile() as render_tf:
        fetch_result = FetchResult(
            Path(fetch_tf.name),
            [RenderError(cjwmodule_i18n_message("http.errors.HttpErrorTimeout"))],
        )
        render_path = Path(render_tf.name)
        result = render(
            pa.table({}),
            P(),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == [cjwmodule_i18n_message("http.errors.HttpErrorTimeout")]
        _assert_table_file(render_path, None)


def test_render_legacy_v0_file():
    with _temp_parquet_file(
        pa.table({"A": [1, 2], "B": ["a", "b"]})
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(first_row_is_header=False),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(
            render_path, pa.table({"A": pa.array([1, 2], pa.int8()), "B": ["a", "b"]})
        )


def test_render_legacy_v0_file_first_row_is_header():
    with _temp_parquet_file(
        pa.table({"1": ["A", "1", "2"], "2": ["B", "a", "b"]})
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(first_row_is_header=True),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        # Auto-converted str to int
        _assert_table_file(
            render_path, pa.table({"A": pa.array([1, 2], pa.int8()), "B": ["a", "b"]})
        )


def test_render_v1():
    with tempfile.NamedTemporaryFile() as fetch_tf, tempfile.NamedTemporaryFile() as render_tf:
        fetch_path = Path(fetch_tf.name)
        render_path = Path(render_tf.name)
        httpfile.write(
            fetch_path,
            {"url": "http://example.org/file"},
            "200 OK",
            [("Content-Type", "text/html; charset=utf-8")],
            io.BytesIO(
                b"""
                    <html>
                        <body>
                            <table>
                                <thead><tr><th>A</th><th>B</th></tr></thead>
                                <tbody>
                                    <tr><td>1</td><td>a</td></tr>
                                    <tr><td>2</td><td>b</td></tr>
                                </tbody>
                            </table>
                        </body>
                    </html>
                """
            ),
        )
        fetch_result = FetchResult(Path(fetch_tf.name), [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", tablenum=1, first_row_is_header=False),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(
            render_path, pa.table({"A": pa.array([1, 2], pa.int8()), "B": ["a", "b"]})
        )


def test_render_v1_missing_charset():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        """
            <table>
                <thead><tr><th>ééééé</th></tr></thead>
                <tbody><tr><td>a</td></tbody>
            </table>
        """.encode(
            "utf-8"
        ),
        headers=[("Content-Type", "text/html")],
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", tablenum=1, first_row_is_header=False),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(render_path, pa.table({"ééééé": ["a"]}))


def test_render_v1_missing_content_type_and_charset():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        """
            <table>
                <thead><tr><th>ééééé</th></tr></thead>
                <tbody><tr><td>a</td></tbody>
            </table>
        """.encode(
            "utf-8"
        ),
        headers=[],
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", tablenum=1, first_row_is_header=False),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(render_path, pa.table({"ééééé": ["a"]}))


def test_render_v1_first_row_is_header():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"""
            <html>
                <body>
                    <table>
                        <thead><tr><th>A</th><th>B</th></tr></thead>
                        <tbody>
                            <tr><td>1</td><td>a</td></tr>
                            <tr><td>2</td><td>b</td></tr>
                        </tbody>
                    </table>
                </body>
            </html>
        """,
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", first_row_is_header=True),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(
            render_path, pa.table({"1": pa.array([2], pa.int8()), "a": ["b"]})
        )


def test_render_v1_wrong_tablenum():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"<html><body><table><tr><th>A</th></tr><tr><td>a</td></tr></table></body></html>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", tablenum=2, first_row_is_header=False),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == [i18n_message("params.badTablenum", {"nTables": 1})]
        _assert_table_file(render_path, None)


def test_render_v1_first_row_is_header_zero_rows():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"<html><body><table><tr><th>A</th><th>B</th></tr></table></body></html>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", first_row_is_header=True),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(
            render_path,
            pa.table({"A": pa.array([], pa.utf8()), "B": pa.array([], pa.utf8())}),
        )


def test_render_v1_http_error():
    with _temp_httpfile(
        "http://example.org/file",
        "404 Not Found",
        b"<html><body>Not found</body></html>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", first_row_is_header=True),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == [i18n_message("http.notOk", {"httpStatus": "404 Not Found"})]
        _assert_table_file(render_path, None)


def test_render_v1_no_tables():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"<html><body><p>No tables</p></body></html>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file", first_row_is_header=True),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == [i18n_message("html.noTables")]
        _assert_table_file(render_path, None)


def test_render_v1_empty_table():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"<html><body><table></table></body></html>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file"),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        # BUG: Pandas reports the wrong error message. (Should be "Table is
        # empty")
        assert result == [i18n_message("html.noTables")]
        _assert_table_file(render_path, None)


def test_render_v1_empty_tr():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"<html><body><table><thead><tr></tr></thead><tbody><tr></tr></tbody></table></body></html>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file"),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        # BUG: Pandas reports the wrong error message. (Should be "Table is
        # empty")
        assert result == [i18n_message("html.noTables")]
        _assert_table_file(render_path, None)


def test_render_v1_merge_colspan_headers():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"""
           <table><thead>
             <tr><th colspan="2">Category</th></tr>'
             <tr><th>A</th><th>B</th></tr>"
           </thead><tbody>
             <tr><td>a</td><td>b</td></tr>"
           </tbody></table>
        """,
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file"),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        assert result == []
        _assert_table_file(
            render_path, pa.table({"Category - A": ["a"], "Category - B": ["b"]})
        )


def test_render_v1_no_thead():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"<table><tbody><tr><td>a</td><td>b</td></tr></tbody></table>",
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file"),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        # [adamhooper, 2021-01-26] I don't expect many users will understand
        # why we chose these column names. It's because pandas.read_html()
        # returned this when we first called it.
        #
        # It makes first_row_is_header very confusing: in _this_ sample table,
        # first_row_is_header is sensible. But what about everywhere _else_?
        assert result == [
            cjwmodule_i18n_message(
                "util.colnames.warnings.default",
                {"n_columns": 2, "first_colname": "Column 1"},
            ),
        ]
        _assert_table_file(
            render_path, pa.table({"Column 1": ["a"], "Column 2": ["b"]})
        )


def test_render_v1_icky_unnamed_columns_from_pandas():
    with _temp_httpfile(
        "http://example.org/file",
        "200 OK",
        b"""
            <table><thead>
              <tr><th></th><th>Column 1</th></tr>
            </thead><tbody>
              <tr><td>a</td><td>b</td><td>c</td></tr>
            </tbody></table>
        """,
    ) as fetch_path, tempfile.NamedTemporaryFile() as render_tf:
        render_path = Path(render_tf.name)
        fetch_result = FetchResult(fetch_path, [])
        result = render(
            pa.table({}),
            P(url="http://example.org/file"),
            render_path,
            fetch_result=fetch_result,
            settings=DefaultSettings(),
        )
        _assert_table_file(
            render_path,
            pa.table(
                {
                    "Unnamed: 0": ["a"],
                    "Column 1": ["b"],
                    "Unnamed: 2": ["c"],
                }
            ),
        )
        assert result == []
