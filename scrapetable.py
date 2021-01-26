"""HTML-scraper data loader: pull a <table> from a downloaded file.

FILE FORMAT (v1)
================

One of:

* An empty file (before first run, or in case of HTTP error)
* A cjwmodule.http.httpfile file (v1)
* A Parquet file (v0-compatible)

The Parquet file should be handled using `render_v0()`, for backwards
compatibility. Do not edit the logic in `render_v0()`.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Tuple, Union

import cjwparquet
import cjwparse.csv
import pandas as pd
from cjwmodule.http import HttpError, httpfile
from cjwmodule.i18n import I18nMessage, trans

EMPTY_DATAFRAME = pd.DataFrame()


class RenderError(NamedTuple):
    """Mimics cjworkbench.cjwkernel.types.RenderError

    TODO move this to cjwmodule, so we can reuse it.
    """

    message: I18nMessage
    quick_fixes: List[None] = []


class FetchResult(NamedTuple):
    """Mimics cjworkbench.cjwkernel.types.FetchResult

    TODO move this to cjwmodule, so we can reuse it.
    """

    path: Path
    errors: List[RenderError] = []


class Settings(Protocol):
    MAX_ROWS_PER_TABLE: int
    MAX_COLUMNS_PER_TABLE: int
    MAX_BYTES_PER_VALUE: int
    MAX_BYTES_TEXT_DATA: int
    MAX_BYTES_PER_COLUMN_NAME: int
    MAX_CSV_BYTES: int
    MAX_DICTIONARY_PYLIST_N_BYTES: int
    MIN_DICTIONARY_COMPRESSION_RATIO_PYLIST_N_BYTES: float


def fetch_arrow(
    params: Dict[str, Any],
    secrets: Dict[str, Any],
    last_fetch_result: Optional[Any],
    input_table_parquet_path: Optional[Path],
    output_path: Path,
) -> FetchResult:
    url = params["url"]
    if not url:
        return FetchResult(output_path)  # don't create a version

    try:
        asyncio.run(httpfile.download(params["url"], output_path))
    except HttpError as err:
        output_path.write_bytes(b"")
        return FetchResult(output_path, [RenderError(err.i18n_message)])

    return FetchResult(output_path)


def _merge_colspan_headers(names: List[Union[int, str, Tuple[str]]]) -> List[str]:
    """Turn tuple colnames into strings.

    Pandas `read_html()` returns tuples for column names when scraping tables
    with colspan. Collapse duplicate entries and reformats to be human
    readable. E.g. ('year', 'year') -> 'year' and
    ('year', 'month') -> 'year - month'

    Returned column names may be duplicates. They may be empty or too long.
    """
    ret = []
    for name in names:
        if isinstance(name, tuple):
            # Convert to dict, then call .keys() -- to remove duplicates.
            # (Python dicts return keys in insertion order)
            name = " - ".join(dict((s, None) for s in name).keys())
        elif isinstance(name, int):
            # If first row isn't header and there's no <thead>, table.columns
            # will be an integer index.
            name = ""  # auto-generate column name
        else:
            # it's already a str
            pass
        ret.append(name)
    return ret


def _write_dataframe_as_arrow_table_and_handle_lots_of_edge_cases(
    *,
    table: pd.DataFrame,
    output_path: Path,
    first_row_is_header: bool,
    colnames: List[str],
    settings: Settings
) -> List[I18nMessage]:
    """Convert ugly Pandas DataFrame to sane Arrow file and warnings.

    Features:

    * Cleans column names
    * Truncates using settings.MAX_ROWS_PER_TABLE
    * Truncates using settings.MAX_COLUMNS_PER_TABLE
    * Truncates using settings.MAX_BYTES_PER_VALUE
    * Truncates using settings.MAX_BYTES_TEXT_DATA
    * Uses original headers if first_row_is_header=False. Omits the header row
      (and uses the first row of data as header) if first_row_is_header=True.
    * Auto-converts text to numbers.
    * If first_row_is_header=True but there are no data rows, assume False.
    """
    # cjwparse does all this already. So let's convert to CSV and back to Arrow.

    # Specify the CSV header row, if we want it
    #
    # parse_csv() will not "trust" this header row: it will clean column
    # names, truncate too-long columns, etc. And if we pass
    # header=False, parse_csv() will do all that logic with the _second_
    # row, because it will treat the second row as header.
    if first_row_is_header and len(table):
        header = False
    else:
        header = colnames

    with tempfile.NamedTemporaryFile() as tf:
        table.to_csv(tf.name, encoding="utf-8", index=False, header=header)
        tf.flush()
        return cjwparse.csv.parse_csv(
            Path(tf.name),
            output_path=output_path,
            encoding="utf-8",
            settings=settings,
            delimiter=",",
            has_header=True,
            autoconvert_text_to_numbers=True,
        )


def _render_v0(
    *,
    fetch_result: FetchResult,
    output_path: Path,
    params: Dict[str, Any],
    settings: Settings
) -> List[I18nMessage]:
    with cjwparquet.open_as_mmapped_arrow(fetch_result.path) as arrow_table:
        table = arrow_table.to_pandas(ignore_metadata=True)

    render_errors = _write_dataframe_as_arrow_table_and_handle_lots_of_edge_cases(
        table=table,
        output_path=output_path,
        first_row_is_header=params["first_row_is_header"],
        colnames=table.columns,
        settings=settings,
    )

    return fetch_result.errors + render_errors


def render(
    arrow_table,
    params: Dict[str, Any],
    output_path: Path,
    *,
    fetch_result: FetchResult,
    settings: Settings,
    **kwargs
) -> List[I18nMessage]:
    errors = [e.message for e in fetch_result.errors]

    if fetch_result.path.stat().st_size == 0:
        return errors  # interpreted as either empty result or error

    if cjwparquet.file_has_parquet_magic_number(fetch_result.path):
        return _render_v0(
            fetch_result=fetch_result,
            output_path=output_path,
            params=params,
            settings=settings,
        )

    # Okay, it's a v1 data file

    # We delve into pd.read_html()'s innards, below. Part of that means some
    # first-use initialization.
    pd.io.html._importers()

    with httpfile.read(fetch_result.path) as http:
        if http.status_line != "200 OK":
            errors.append(
                trans(
                    "http.notOk",
                    "Server gave unexpected HTTP response: {httpStatus}",
                    {"httpStatus": http.status_line},
                )
            )
            return errors

        content_type = httpfile.extract_first_header(http.headers, "Content-Type")
        if content_type:
            try:
                encoding = content_type.split("charset=")[1]
            except IndexError:
                encoding = None
        else:
            encoding = None

        try:
            tables = pd.io.html._parse(
                # Positional arguments:
                flavor="html5lib",  # force algorithm, for reproducibility
                io=http.body_path.as_posix(),
                encoding=encoding,
                match=".+",
                attrs=None,
                displayed_only=False,  # avoid dud feature: it ignores CSS
                # Required kwargs that pd.read_html() would set by default:
                header=None,
                skiprows=None,
                # Now the reason we used pd.io.html._parse() instead of
                # pd.read_html(): we get to pass whatever kwargs we want to
                # TextParser.
                #
                # kwargs we get to add as a result of this hack:
                na_filter=False,  # do not autoconvert
                dtype=str,  # do not autoconvert
            )
        except ValueError:
            # pandas.read_html() gives this unhelpful error message....
            errors.append(
                trans("html.noTables", "Did not find any <table> tags in this page")
            )
            return errors
        except IndexError:
            # pandas.read_html() gives this unhelpful error message....
            # TODO reproduce this in our test suite
            errors.append(
                trans("html.noColumns", "Found a <table> tag with no columns")
            )
            return errors

    if not len(tables):
        # TODO reproduce this in our test suite
        errors.append(
            trans("html.noTables", "Did not find any <table> tags in this page")
        )
        return errors

    if params["tablenum"] <= 0 or params["tablenum"] > len(tables):
        errors.append(
            trans(
                "params.badTablenum",
                "Please choose a table position between 1 and {nTables}",
                {"nTables": len(tables)},
            )
        )
        return errors

    table = tables[params["tablenum"] - 1]

    errors.extend(
        _write_dataframe_as_arrow_table_and_handle_lots_of_edge_cases(
            table=table,
            output_path=output_path,
            colnames=_merge_colspan_headers(table.columns),
            first_row_is_header=params["first_row_is_header"],
            settings=settings,
        )
    )

    return errors
