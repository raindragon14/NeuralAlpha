"""Raw data fetching from external sources.

Sources:
- Stock prices ``.JK``: Yahoo Finance.
- USD/IDR exchange rate (JISDOR) and BI-7DRRR: Bank Indonesia.

The functions here return the data as-is; cleaning is done at the next stage
(``02_build_features.py``).
"""

from __future__ import annotations

import io
import logging
import re
import time
import zipfile
from collections.abc import Sequence
from xml.etree import ElementTree

import pandas as pd
import requests

BI_RATE_URL = "https://www.bi.go.id/en/statistik/indikator/bi-rate.aspx"
BI_JISDOR_URL = "https://www.bi.go.id/id/statistik/informasi-kurs/jisdor/default.aspx"
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124 Safari/537.36"
)

_logger = logging.getLogger(__name__)


def _session() -> requests.Session:
    """Create an HTTP session with a browser User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})
    return session


def _request(session: requests.Session, method: str, **kwargs) -> requests.Response:
    """HTTP request with simple retries."""
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            return getattr(session, method)(**kwargs)
        except requests.RequestException as exc:  # the BI site often drops connections
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed after 4 attempts: {last_error}")


def fetch_equities(
    tickers: Sequence[str],
    start: str,
    end: str,
    retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV per ticker via Yahoo Finance.

    ``auto_adjust=False`` keeps ``Adj Close`` separate from ``Close``. Tickers
    without data (for example delisted ones) are mapped to ``None``.
    """
    import yfinance as yf

    results: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        frame: pd.DataFrame | None = None
        for attempt in range(retries):
            try:
                frame = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                break
            except Exception as exc:  # yfinance raises several unrelated types
                if attempt == retries - 1:
                    raise
                _logger.warning(
                    "download failed for %s (attempt %d/%d): %s",
                    ticker,
                    attempt + 1,
                    retries,
                    exc,
                )
                time.sleep(2)
        if frame is None or frame.empty:
            results[ticker] = None
            continue
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame.index.name = "Date"
        results[ticker] = frame
        time.sleep(0.4)
    return results


def _xlsx_rows(content: bytes) -> list[dict[str, str]]:
    """Read the first sheet's rows of an ``.xlsx`` without extra libraries.

    Only handles single-sheet exports from Bank Indonesia: shared strings and
    inline numbers, without formulas.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{_XLSX_NS}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{_XLSX_NS}t")))
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))

    rows: list[dict[str, str]] = []
    for row in sheet.iter(f"{_XLSX_NS}row"):
        cell_values: dict[str, str] = {}
        for c in row.findall(f"{_XLSX_NS}c"):
            ref = c.get("r", "")
            match = re.match(r"[A-Z]+", ref)
            if match is None:
                continue
            column = match.group(0)
            value = c.find(f"{_XLSX_NS}v")
            text = value.text if value is not None else None
            if c.get("t") == "s" and text is not None:
                cell_values[column] = shared[int(text)]
            else:
                cell_values[column] = text or ""
        rows.append(cell_values)
    return rows


def fetch_jisdor(start: str, end: str) -> pd.DataFrame:
    """Download the official JISDOR reference rate (USD/IDR) from Bank Indonesia.

    The BI site provides a date-range selector and an export button; this
    function mimics that request and then reads the resulting ``.xlsx`` file.
    Output columns: ``date`` and ``rate`` (rupiah per 1 USD).
    """
    session = _session()
    html = _request(session, "get", url=BI_JISDOR_URL, timeout=25).text

    prefix_match = re.search(r'name="([^"]*TextBoxFrom)"', html)
    if prefix_match is None:
        raise RuntimeError("JISDOR date-range control not found")
    prefix = prefix_match.group(1).rsplit("TextBoxFrom", 1)[0]

    start_text = pd.Timestamp(start).strftime("%d/%m/%Y")
    end_text = pd.Timestamp(end).strftime("%d/%m/%Y")
    payload = _bi_hidden(html)
    payload[prefix + "TextBoxFrom"] = start_text
    payload[prefix + "TextBoxDateTo"] = end_text
    payload[prefix + "HiddenFieldDateFrom"] = start_text
    payload[prefix + "HiddenFieldDateTo"] = end_text
    payload[prefix + "ButtonExport"] = "Unduh"

    response = _request(session, "post", url=BI_JISDOR_URL, data=payload, timeout=120)
    if response.content[:2] != b"PK":
        raise RuntimeError("JISDOR export is not an .xlsx file")

    records: list[dict[str, object]] = []
    for cell in _xlsx_rows(response.content):
        date_text = cell.get("B", "")
        rate_text = cell.get("C", "")
        if not date_text or not rate_text:
            continue
        timestamp = pd.to_datetime(
            date_text, format="%m/%d/%Y %I:%M:%S %p", errors="coerce"
        )
        if pd.isna(timestamp):
            continue
        try:
            value = float(rate_text)
        except ValueError:
            continue
        records.append({"date": timestamp, "rate": value})
    result = pd.DataFrame(records).dropna(subset=["date"])
    return result.sort_values("date").reset_index(drop=True)


def _bi_table(html: str) -> pd.DataFrame:
    """Read the first table from the Bank Indonesia page HTML."""
    tables = pd.read_html(io.StringIO(html))
    frame = tables[0]
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _bi_next_target(html: str) -> str | None:
    """Name of the "Next" button in the DataPager while it is still active."""
    for match in re.finditer(r"<input type=\"image\"[^>]*>", html):
        tag = match.group(0)
        if "DataPagerBI7DRR" in tag and "next" in tag and "disabled" not in tag:
            name = re.search(r'name="([^"]+)"', tag)
            if name is not None:
                return name.group(1)
    return None


def _bi_hidden(html: str) -> dict[str, str]:
    """Collect all hidden inputs of the ASP.NET form."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r'<input type="hidden"[^>]*name="([^"]+)"[^>]*value="([^"]*)"', html
        )
    }


def fetch_bi_rate(max_pages: int = 40) -> pd.DataFrame:
    """Fetch the full history of BI-7DRRR decisions (the reference rate).

    The indicator page shows 10 meetings per page; this function walks all
    pages and then returns the ``date`` and ``rate`` columns (in percent).
    """
    session = _session()
    response = _request(session, "get", url=BI_RATE_URL, timeout=25)
    html = response.text

    records: dict[int, tuple[str, float]] = {}
    for _ in range(max_pages):
        frame = _bi_table(html)
        for _, row in frame.iterrows():
            records[int(row.iloc[0])] = (
                str(row.iloc[1]),
                float(str(row.iloc[2]).replace("%", "").strip()),
            )
        target = _bi_next_target(html)
        if target is None:
            break
        payload = _bi_hidden(html)
        payload[f"{target}.x"] = "1"
        payload[f"{target}.y"] = "1"
        payload["__EVENTTARGET"] = ""
        payload["__EVENTARGUMENT"] = ""
        html = _request(session, "post", url=BI_RATE_URL, data=payload, timeout=30).text
        time.sleep(0.6)

    if not records:
        raise RuntimeError("BI-7DRRR table could not be read")

    # Sequence numbers decrease over time; sort ascending by date.
    rows = [
        {
            "date": pd.to_datetime(period, format="%d %B %Y"),
            "rate": rate,
        }
        for period, rate in records.values()
    ]
    result = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return result
