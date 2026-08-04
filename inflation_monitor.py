from __future__ import annotations

import calendar
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import pickle
from dataclasses import dataclass, field
from io import BytesIO, StringIO
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urljoin
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import urllib3
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


APP_TITLE = "Inflation Monitor"
APP_SUBTITLE = (
    "Official-source inflation dashboard focused on US CPI, validated leading signals, "
    "and transparent source QA."
)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
NBS_ARCHIVE_URL = "https://www.stats.gov.cn/english/PressRelease/"
NBS_SEARCH_URL = "https://www.stats.gov.cn/search/english/s"
NBS_TITLE_PREFIX = "Industrial Producer Price Indexes in "
NBS_PPI_TITLE_VARIANTS = (
    "industrial producer price indexes in ",
    "producer price index in the industrial sector for ",
    "producer prices in the industrial sector for ",
    "producer prices for the industrial sector for ",
)
NBS_SEARCH_QUERIES = (
    "Industrial Producer Price Indexes in",
    "Producer Price Index in the Industrial Sector",
    "Producer Prices in the Industrial Sector",
    "Producer Prices for the Industrial Sector",
)
ISM_SITEMAP_URL = "https://www.ismworld.org/sitemap.xml"
ISM_DIRECT_REPORT_ROOT = (
    "https://www.ismworld.org/supply-management-news-and-reports/"
    "reports/ism-pmi-reports/pmi/"
)
FAO_PAGE_URL = "https://www.fao.org/worldfoodsituation/foodpricesindex/en/"
CLEVELAND_MONTH_JSON_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_month.json?sc_lang=en"
)
CLEVELAND_QUARTER_JSON_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/"
    "inflationnowcasting/nowcast_quarter.json?sc_lang=en"
)
ATLANTA_WAGE_GROWTH_URL = (
    "https://www.atlantafed.org/research-and-data/data/wage-growth-tracker"
)
ATLANTA_STICKY_CPI_URL = (
    "https://www.atlantafed.org/research-and-data/data/sticky-price-cpi"
)

PLOTLY_CONFIG = {"displaylogo": False, "responsive": True}
MONTHLY_CACHE_TTL_SECONDS = 60 * 60 * 6
CLEVELAND_CACHE_TTL_SECONDS = 60 * 60
DEFAULT_TIMEOUT_SECONDS = 40
NETWORK_RETRY_ATTEMPTS = 2
NETWORK_RETRY_BACKOFF_SECONDS = 1.5
NBS_MAX_WORKERS = 6
FRED_MAX_WORKERS = 8
SNAPSHOT_DIR = Path(".cache")
MONTHLY_SNAPSHOT_PATH = SNAPSHOT_DIR / "inflation_monitor_monthly_payload.pkl"
CLEVELAND_SNAPSHOT_PATH = SNAPSHOT_DIR / "inflation_monitor_cleveland_payload.pkl"
MONTHLY_SNAPSHOT_MAX_AGE_SECONDS = 60 * 60 * 12
CLEVELAND_SNAPSHOT_MAX_AGE_SECONDS = 60 * 60 * 2

FRESHNESS_RULES = {
    "cleveland_nowcast_month": 7,
    "cleveland_nowcast_quarter": 7,
    "case_shiller": 120,
}
DEFAULT_MONTHLY_FRESHNESS_DAYS = 60

FRED_SERIES = {
    "headline_cpi": ("CPIAUCSL", "US CPI"),
    "core_cpi": ("CPILFESL", "US Core CPI"),
    "headline_ppi": ("PPIFIS", "US PPI Final Demand"),
    "core_ppi": ("PPIFES", "US PPI Final Demand ex Food and Energy"),
    "china_import_price": ("CHNTOT", "US Import Price Index from China"),
    "used_vehicle_ppi": ("PCU441110441110102", "PPI Used Vehicle Sales"),
    "used_car_cpi": ("CUSR0000SETA02", "CPI Used Cars and Trucks"),
    "wti_crude_oil": ("MCOILWTICO", "WTI Crude Oil Spot Price"),
    "motor_fuel_cpi": ("CUSR0000SETB01", "CPI Motor Fuel"),
    "henry_hub_gas": ("MHHNGSP", "Henry Hub Natural Gas Spot Price"),
    "utility_gas_cpi": ("CUSR0000SEHF02", "CPI Utility Gas Service"),
    "apparel_cpi": ("CPIAPPSL", "CPI Apparel"),
    "case_shiller": ("CSUSHPINSA", "Case-Shiller National Home Price Index"),
    "shelter_cpi": ("CUSR0000SAH1", "CPI Shelter"),
    "food_home_cpi": ("CUSR0000SAF11", "CPI Food at Home"),
}

DISPLAY_LABELS = {
    "headline_cpi": "Headline CPI",
    "core_cpi": "Core CPI",
    "headline_ppi": "Headline PPI",
    "core_ppi": "Core PPI",
    "china_import_price": "Import Prices from China",
    "used_vehicle_ppi": "Used Vehicle PPI",
    "used_car_cpi": "Used Car CPI",
    "wti_crude_oil": "WTI Crude Oil",
    "motor_fuel_cpi": "Motor Fuel CPI",
    "henry_hub_gas": "Henry Hub Gas",
    "utility_gas_cpi": "Utility Gas CPI",
    "apparel_cpi": "Apparel CPI",
    "case_shiller": "Case-Shiller HPI",
    "shelter_cpi": "Shelter CPI",
    "food_home_cpi": "Food-at-Home CPI",
    "fao_food_price_index": "FAO Food Price Index",
    "atlanta_sticky_cpi": "Atlanta Sticky CPI",
    "atlanta_wage_growth": "Atlanta Wage Growth",
    "china_ppi_yoy": "China PPI YoY",
    "ism_prices_paid": "ISM Prices Paid",
}

MONTH_TO_NUMBER = {
    month.lower(): index
    for index, month in enumerate(calendar.month_name)
    if month
}

ISM_URL_PATTERN = re.compile(
    r"report-on-business-roundup-"
    r"(?P<month>[a-z]+)"
    r"(?:-(?P<year>\d{4}))?"
    r"-manufacturing-pmi",
    re.IGNORECASE,
)


@dataclass
class SourceResult:
    key: str
    label: str
    frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_url: str = ""
    ok: bool = False
    detail: str = ""
    latest_observation: pd.Timestamp | None = None
    fetched_at: pd.Timestamp | None = None
    freshness_days: int | None = None
    stale: bool = False
    insecure_tls: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def row_count(self) -> int:
        return int(len(self.frame))


@dataclass(frozen=True)
class IndicatorSpec:
    key: str
    title: str
    indicator_key: str
    target_key: str
    indicator_transform: str = "raw"
    target_transform: str = "yoy"
    lag_min: int = 0
    lag_max: int = 0
    note: str = ""
    secondary_axis: bool = False


@dataclass
class ValidationReport:
    key: str
    title: str
    indicator_key: str
    target_key: str
    indicator_label: str
    target_label: str
    approved: bool
    reasons: list[str] = field(default_factory=list)
    lag_months: int | None = None
    train_best_lag: int | None = None
    holdout_best_lag: int | None = None
    full_corr: float | None = None
    holdout_corr: float | None = None
    rolling_corr_24: float | None = None
    overlap: int = 0
    freshness_ok: bool = False
    duplicate_months: bool = False
    raw_indicator: pd.DataFrame = field(default_factory=pd.DataFrame)
    target_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    aligned_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    shifted_indicator_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    projection_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    note: str = ""
    secondary_axis: bool = False
    source_urls: dict[str, str] = field(default_factory=dict)
    latest_indicator_date: pd.Timestamp | None = None
    latest_target_date: pd.Timestamp | None = None
    latest_zscore: float | None = None
    stale_gap_days: int | None = None
    freshness_note: str = ""
    pass_through_beta: float | None = None
    pass_through_intercept: float | None = None
    score_weight: float | None = None


LEADING_SPECS: tuple[IndicatorSpec, ...] = (
    IndicatorSpec(
        key="used_vehicle_ppi_to_used_car_cpi",
        title="Used Vehicle PPI -> Used Car CPI",
        indicator_key="used_vehicle_ppi",
        target_key="used_car_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="Used-vehicle producer prices often move before used-car CPI.",
    ),
    IndicatorSpec(
        key="fao_to_food_home",
        title="FAO FFPI -> Food-at-Home CPI",
        indicator_key="fao_food_price_index",
        target_key="food_home_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="Global food input prices can feed into grocery inflation with a short lag.",
    ),
    IndicatorSpec(
        key="wti_to_motor_fuel",
        title="WTI Crude Oil -> Motor Fuel CPI",
        indicator_key="wti_crude_oil",
        target_key="motor_fuel_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="Monthly crude prices are tested as an upstream lead for motor-fuel inflation.",
    ),
    IndicatorSpec(
        key="henry_hub_to_utility_gas",
        title="Henry Hub Gas -> Utility Gas CPI",
        indicator_key="henry_hub_gas",
        target_key="utility_gas_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="Natural-gas spot prices are tested against regulated utility-gas CPI with a short lag.",
    ),
    IndicatorSpec(
        key="china_ppi_to_apparel_cpi",
        title="China PPI -> Apparel CPI",
        indicator_key="china_ppi_yoy",
        target_key="apparel_cpi",
        indicator_transform="raw",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="China factory-gate prices are tested against US apparel CPI as a China-exposed goods channel.",
    ),
    IndicatorSpec(
        key="case_shiller_to_shelter",
        title="Case-Shiller -> Shelter CPI",
        indicator_key="case_shiller",
        target_key="shelter_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=9,
        lag_max=18,
        note="Housing-price momentum usually reaches shelter CPI with a long lag.",
    ),
    IndicatorSpec(
        key="headline_ppi_to_headline_cpi",
        title="Headline PPI -> Headline CPI",
        indicator_key="headline_ppi",
        target_key="headline_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=12,
        note="Producer-price pressure is tested as a leading signal for headline CPI.",
    ),
    IndicatorSpec(
        key="core_ppi_to_core_cpi",
        title="Core PPI -> Core CPI",
        indicator_key="core_ppi",
        target_key="core_cpi",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=12,
        note="Core producer prices are tested as a lead for core CPI.",
    ),
    IndicatorSpec(
        key="ism_prices_paid_to_headline_ppi",
        title="ISM Prices Paid -> Headline PPI",
        indicator_key="ism_prices_paid",
        target_key="headline_ppi",
        indicator_transform="raw",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="ISM manufacturing prices-paid survey is a classic early read on producer-price pressure.",
    ),
    IndicatorSpec(
        key="china_ppi_to_import_prices",
        title="China PPI -> Import Prices from China",
        indicator_key="china_ppi_yoy",
        target_key="china_import_price",
        indicator_transform="raw",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
        note="China factory-gate prices are tested against US import prices from China.",
    ),
)

WATCHLIST_KEYS = frozenset(
    {
        "headline_ppi_to_headline_cpi",
        "core_ppi_to_core_cpi",
    }
)

MONTHLY_SOURCE_KEYS = frozenset(
    {
        *FRED_SERIES.keys(),
        "fao_food_price_index",
        "atlanta_wage_growth",
        "atlanta_sticky_cpi",
        "china_ppi_yoy",
        "ism_prices_paid",
    }
)
CLEVELAND_SOURCE_KEYS = frozenset({"cleveland_nowcast_month", "cleveland_nowcast_quarter"})


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _slugify(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        flattened = []
        for index, values in enumerate(data.columns):
            tokens = [str(value).strip() for value in values if str(value).strip()]
            name = " ".join(token for token in tokens if "unnamed" not in token.lower())
            flattened.append(name or f"column_{index}")
        data.columns = flattened
    else:
        data.columns = [str(column).strip() for column in data.columns]
    return data


def _clean_series_values(series: pd.Series) -> pd.Series:
    return series.replace({".": np.nan, "na": np.nan, "NA": np.nan, "": np.nan})


def _empty_monthly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "value", "release_date"])


def _coerce_datetime(series: pd.Series) -> pd.Series:
    cleaned = _clean_series_values(series)
    try:
        return pd.to_datetime(cleaned, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(cleaned, errors="coerce")


def _normalize_monthly_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return _empty_monthly_frame()

    data = raw.copy()
    if "date" not in data.columns or "value" not in data.columns:
        raise ValueError("Expected date/value columns.")

    data["date"] = _coerce_datetime(data["date"]).dt.to_period("M").dt.to_timestamp()
    data["value"] = pd.to_numeric(_clean_series_values(data["value"]), errors="coerce")
    if "release_date" not in data.columns:
        data["release_date"] = pd.NaT
    data["release_date"] = _coerce_datetime(data["release_date"])
    data = data.dropna(subset=["date", "value"]).copy()
    if data.empty:
        return _empty_monthly_frame()

    data = data.sort_values(["date", "release_date"], ascending=[True, False])
    data = data.drop_duplicates(subset=["date"], keep="first")
    return data[["date", "value", "release_date"]].reset_index(drop=True)


def _freshness_days_for_key(source_key: str) -> int:
    return int(FRESHNESS_RULES.get(source_key, DEFAULT_MONTHLY_FRESHNESS_DAYS))


def _finalize_source_result(
    key: str,
    label: str,
    frame: pd.DataFrame,
    source_url: str,
    *,
    insecure_tls: bool = False,
    detail: str = "",
    metadata: dict[str, Any] | None = None,
    latest_override: pd.Timestamp | None = None,
) -> SourceResult:
    latest_observation = latest_override
    if latest_observation is None and not frame.empty:
        if "date" in frame.columns:
            latest_observation = pd.to_datetime(frame["date"], errors="coerce").dropna().max()

    freshness_reference = latest_override
    if freshness_reference is None and not frame.empty:
        if "release_date" in frame.columns:
            release_max = _coerce_datetime(frame["release_date"]).dropna().max()
            if pd.notna(release_max):
                freshness_reference = pd.Timestamp(release_max)
        if freshness_reference is None and latest_observation is not None:
            freshness_reference = pd.Timestamp(latest_observation) + pd.offsets.MonthEnd(1)

    freshness_days = _freshness_days_for_key(key)
    stale = False
    if freshness_reference is not None:
        stale = (_utc_now().normalize() - pd.Timestamp(freshness_reference).normalize()).days > freshness_days

    payload_metadata = dict(metadata or {})
    if freshness_reference is not None:
        payload_metadata["freshness_reference"] = _serialize_timestamp(pd.Timestamp(freshness_reference))

    return SourceResult(
        key=key,
        label=label,
        frame=frame,
        source_url=source_url,
        ok=not frame.empty,
        detail=detail or ("loaded" if not frame.empty else "empty"),
        latest_observation=latest_observation,
        fetched_at=_utc_now(),
        freshness_days=freshness_days,
        stale=stale,
        insecure_tls=insecure_tls,
        metadata=payload_metadata,
    )


def _new_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def _request(
    session: requests.Session,
    url: str,
    *,
    method: str = "GET",
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[requests.Response, bool]:
    verify = True
    last_exc: Exception | None = None
    for attempt in range(NETWORK_RETRY_ATTEMPTS):
        try:
            response = session.request(
                method,
                url,
                params=params,
                data=data,
                headers=headers,
                timeout=timeout,
                verify=verify,
            )
            response.raise_for_status()
            return response, not verify
        except requests.exceptions.SSLError:
            if verify:
                # A cert-validation failure isn't a transient network issue;
                # fall back to an insecure request rather than burning a retry.
                verify = False
                continue
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            last_exc = exc
            if attempt < NETWORK_RETRY_ATTEMPTS - 1:
                time.sleep(NETWORK_RETRY_BACKOFF_SECONDS * (attempt + 1))
                continue
            raise
    raise last_exc  # pragma: no cover - unreachable, loop always returns or raises


def _request_text(
    session: requests.Session,
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[str, bool]:
    response, insecure_tls = _request(session, url, timeout=timeout)
    return response.text, insecure_tls


def _request_bytes(
    session: requests.Session,
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> tuple[bytes, bool]:
    response, insecure_tls = _request(session, url, timeout=timeout)
    return response.content, insecure_tls


def _extract_download_href_from_html(
    html: str,
    *,
    page_url: str,
    label_pattern: str = r"Download the data|Download Data",
) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True)
        if re.search(label_pattern, text, re.IGNORECASE):
            return urljoin(page_url, anchor["href"])
    raise ValueError(f"Download link not found on {page_url}")


def _extract_fao_csv_href_from_html(html: str, *, page_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True)
        if "CSV:" in text and "Nominal indices" in text:
            return urljoin(page_url, anchor["href"])
    raise ValueError("FAO CSV link not found.")


def _build_date_series(frame: pd.DataFrame) -> pd.Series | None:
    data = _flatten_columns(frame)
    for column in data.columns:
        if _slugify(column) in {"date", "month", "period"}:
            parsed = _coerce_datetime(data[column])
            if parsed.notna().mean() >= 0.7:
                return parsed

    first_column = data.columns[0]
    parsed = _coerce_datetime(data[first_column])
    if parsed.notna().mean() >= 0.7:
        return parsed
    return None


def _extract_series_from_book(
    book: dict[str, pd.DataFrame],
    *,
    include_tokens: tuple[str, ...],
    exclude_tokens: tuple[str, ...] = (),
) -> pd.DataFrame:
    best_score = float("-inf")
    best_frame = _empty_monthly_frame()

    for _, raw_frame in book.items():
        data = _flatten_columns(raw_frame).dropna(how="all").dropna(axis=1, how="all")
        if data.empty:
            continue

        date_series = _build_date_series(data)
        if date_series is None:
            continue

        for column in data.columns:
            if column == data.columns[0]:
                continue
            slug = _slugify(column)
            if any(token in slug for token in exclude_tokens):
                continue
            numeric = pd.to_numeric(_clean_series_values(data[column]), errors="coerce")
            if numeric.notna().mean() < 0.5:
                continue

            score = 0.0
            score += sum(4.0 for token in include_tokens if token in slug)
            score -= sum(5.0 for token in exclude_tokens if token in slug)
            score += float(numeric.notna().sum()) / 1000.0
            if score <= best_score:
                continue

            candidate = pd.DataFrame({"date": date_series, "value": numeric})
            candidate = candidate.dropna(subset=["date", "value"])
            if candidate.empty:
                continue

            best_score = score
            best_frame = _normalize_monthly_frame(candidate)

    return best_frame


def fetch_fred_series(
    session: requests.Session,
    *,
    key: str,
    series_id: str,
    label: str,
) -> SourceResult:
    url = FRED_GRAPH_URL.format(series_id=series_id)
    text, insecure_tls = _request_text(session, url)
    data = pd.read_csv(StringIO(text))
    data = data.rename(columns={data.columns[0]: "date", data.columns[1]: "value"})
    frame = _normalize_monthly_frame(data[["date", "value"]])
    return _finalize_source_result(
        key,
        label,
        frame,
        f"https://fred.stlouisfed.org/series/{series_id}",
        insecure_tls=insecure_tls,
    )


def fetch_fao_food_price_index(session: requests.Session) -> SourceResult:
    html, page_insecure = _request_text(session, FAO_PAGE_URL)
    csv_url = _extract_fao_csv_href_from_html(html, page_url=FAO_PAGE_URL)
    csv_text, csv_insecure = _request_text(session, csv_url)
    data = pd.read_csv(StringIO(csv_text), skiprows=2)
    data = data.rename(columns={"Date": "date", "Food Price Index": "value"})
    frame = _normalize_monthly_frame(data[["date", "value"]])
    return _finalize_source_result(
        "fao_food_price_index",
        "FAO Food Price Index",
        frame,
        csv_url,
        insecure_tls=page_insecure or csv_insecure,
    )


def fetch_atlanta_wage_growth(session: requests.Session) -> SourceResult:
    html, page_insecure = _request_text(session, ATLANTA_WAGE_GROWTH_URL)
    file_url = _extract_download_href_from_html(html, page_url=ATLANTA_WAGE_GROWTH_URL)
    content, file_insecure = _request_bytes(session, file_url, timeout=90)
    book = pd.read_excel(BytesIO(content), sheet_name=None)
    frame = _extract_series_from_book(
        book,
        include_tokens=("overall",),
        exclude_tokens=("non smoothed", "recession"),
    )
    return _finalize_source_result(
        "atlanta_wage_growth",
        "Atlanta Fed Wage Growth Tracker",
        frame,
        file_url,
        insecure_tls=page_insecure or file_insecure,
    )


def fetch_atlanta_sticky_cpi(session: requests.Session) -> SourceResult:
    html, page_insecure = _request_text(session, ATLANTA_STICKY_CPI_URL)
    file_url = _extract_download_href_from_html(html, page_url=ATLANTA_STICKY_CPI_URL)
    content, file_insecure = _request_bytes(session, file_url, timeout=90)
    book = pd.read_excel(BytesIO(content), sheet_name=None)
    data = _flatten_columns(book["Data"])
    sticky_index = None
    columns = list(data.columns)
    for index, column in enumerate(columns):
        if _slugify(column) == "sticky cpi monthly":
            sticky_index = index
            break
    if sticky_index is None or sticky_index + 3 >= len(columns):
        raise ValueError("Sticky CPI 12-month column not found.")

    frame = pd.DataFrame(
        {
            "date": _coerce_datetime(data[columns[0]]),
            "value": pd.to_numeric(_clean_series_values(data[columns[sticky_index + 3]]), errors="coerce"),
        }
    )
    frame = _normalize_monthly_frame(frame)
    return _finalize_source_result(
        "atlanta_sticky_cpi",
        "Atlanta Fed Sticky CPI",
        frame,
        file_url,
        insecure_tls=page_insecure or file_insecure,
    )


def _parse_observation_month(text: str) -> pd.Timestamp:
    match = re.search(r"([A-Za-z]+)\s+(\d{4})", text)
    if not match:
        raise ValueError(f"Could not parse observation month from: {text}")
    month_number = MONTH_TO_NUMBER[match.group(1).lower()]
    return pd.Timestamp(year=int(match.group(2)), month=month_number, day=1)


def _signed_value(verb: str, value: float | None) -> float:
    normalized = verb.strip().lower()
    if "unchanged" in normalized or "flat" in normalized:
        return 0.0
    if value is None:
        raise ValueError("Directional value missing.")
    negative_tokens = ("decrease", "declin", "fell", "drop", "down")
    return -float(value) if any(token in normalized for token in negative_tokens) else float(value)


def _is_nbs_ppi_title(title: str) -> bool:
    normalized = re.sub(r"^\d+\.\s*", "", title.strip()).lower()
    return any(token in normalized for token in NBS_PPI_TITLE_VARIANTS)


def _normalize_nbs_metric_text(text: str) -> str:
    normalized = BeautifulSoup(text or "", "html.parser").get_text(" ", strip=True)
    normalized = normalized.replace("\xa0", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"(\d+)\s+(\d+)(?=\s+(?:percent|%))", r"\1.\2", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _parse_metric_number(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = str(value).strip().replace(" ", ".")
    return float(normalized) if normalized else None


def _try_parse_nbs_paragraph_metric(text: str, mode: str) -> float | None:
    text = _normalize_nbs_metric_text(text)
    metric_pattern = r"\d+(?:[.\s]\d+)?"
    yoy_pattern = r"year[- ]on(?:[- ]year)?"
    mom_pattern = r"month[- ]on(?:[- ]month)?"
    if mode == "yoy":
        patterns = (
            re.compile(
                r"producer price index for industrial products \(ppi\)\s+"
                r"(?P<verb>decreased|declined|fell|dropped|increased|rose|grew|"
                r"remained unchanged|remained flat|was flat)"
                rf"(?:\s+by\s+(?P<value>{metric_pattern}))?(?:\s+percent|%)?\s+{yoy_pattern}",
                re.IGNORECASE,
            ),
            re.compile(
                r"producer price index(?:\s*\(ppi\)|\s+ppi)? for manufactured goods\s+"
                r"(?P<verb>decreased|declined|fell|dropped|increased|rose|grew|"
                r"remained unchanged|remained flat|was flat)"
                rf"(?:\s+by\s+(?P<value>{metric_pattern}))?(?:\s+percent|%)?\s+{yoy_pattern}",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:the national\s+)?producer prices? for industrial products\s+"
                r"(?P<verb>went up|went down|decreased|declined|fell|dropped|increased|rose|grew|"
                r"remained unchanged|remained flat|was flat)"
                rf"(?:\s+by\s+(?P<value>{metric_pattern}))?(?:\s+percent|%)?\s+{yoy_pattern}",
                re.IGNORECASE,
            ),
        )
    else:
        patterns = (
            re.compile(
                r"(?:it|and|and it|the producer price index for industrial products)?\s*"
                r"(?P<verb>decreased|declined|fell|dropped|increased|rose|grew|"
                r"remained unchanged|remained flat|was flat)"
                rf"(?:\s+by\s+(?P<value>{metric_pattern}))?(?:\s+percent|%)?\s+{mom_pattern}",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:it|and|and it|producer price index(?:\s*\(ppi\)|\s+ppi)? for manufactured goods)?\s*"
                r"(?P<verb>decreased|declined|fell|dropped|increased|rose|grew|"
                r"remained unchanged|remained flat|was flat)"
                rf"(?:\s+by\s+(?P<value>{metric_pattern}))?(?:\s+percent|%)?\s+{mom_pattern}",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:the national\s+)?producer prices? for industrial products\s+"
                r"(?P<verb>went up|went down|decreased|declined|fell|dropped|increased|rose|grew|"
                r"remained unchanged|remained flat|was flat)"
                rf"(?:\s+by\s+(?P<value>{metric_pattern}))?(?:\s+percent|%)?\s+{mom_pattern}",
                re.IGNORECASE,
            ),
            re.compile(
                r"producer price index(?:\s*\(ppi\)|\s+ppi)? for manufactured goods\s+"
                rf"(?:decreased|declined|fell|dropped|increased|rose|grew|went up|went down)\s+by\s+{metric_pattern}"
                rf"(?:\s+percent|%)?\s+{yoy_pattern}\s+and\s+(?P<value>"
                rf"{metric_pattern})"
                rf"(?:\s+percent|%)?\s+{mom_pattern}",
                re.IGNORECASE,
            ),
        )

    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        value = match.group("value")
        if "verb" not in pattern.groupindex:
            return _parse_metric_number(value)
        return _signed_value(match.group("verb"), _parse_metric_number(value))
    return None


def _try_parse_nbs_table_values(text: str) -> tuple[float | None, float | None]:
    table_match = re.search(
        r"I\.\s*Producer Price Indexes for Industrial Products\s+"
        r"([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)\s+([+-]?\d+(?:\.\d+)?)",
        text,
        re.IGNORECASE,
    )
    if not table_match:
        return None, None
    mom = float(table_match.group(1))
    yoy = float(table_match.group(2))
    return yoy, mom


def _parse_nbs_article_html(
    html: str,
    *,
    article_url: str,
    title_hint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.find(["h1", "h2"])
    title_text = title_node.get_text(" ", strip=True) if title_node else title_hint
    text = _normalize_nbs_metric_text(" ".join(soup.stripped_strings))

    observation_date = _parse_observation_month(re.sub(r"^\d+\.\s*", "", title_text))
    release_match = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    release_date = pd.to_datetime(release_match.group(1)) if release_match else pd.NaT

    yoy = _try_parse_nbs_paragraph_metric(text, "yoy")
    mom = _try_parse_nbs_paragraph_metric(text, "mom")
    if yoy is None or mom is None:
        table_yoy, table_mom = _try_parse_nbs_table_values(text)
        yoy = table_yoy if yoy is None else yoy
        mom = table_mom if mom is None else mom

    if yoy is None or mom is None:
        raise ValueError(f"Could not parse NBS values from {article_url}")

    common = {"date": observation_date, "release_date": release_date}
    return ({"value": yoy, **common}, {"value": mom, **common})


def _fetch_nbs_archive_page(page_number: int) -> tuple[int, list[tuple[str, str]], bool]:
    session = _new_session()
    suffix = "index.html" if page_number == 1 else f"index_{page_number}.html"
    page_url = urljoin(NBS_ARCHIVE_URL, suffix)
    try:
        response, insecure_tls = _request(session, page_url)
    except requests.RequestException:
        return page_number, [], False

    soup = BeautifulSoup(response.text, "html.parser")
    page_links: list[tuple[str, str]] = []
    seen_page: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        title = anchor.get_text(" ", strip=True)
        if not _is_nbs_ppi_title(title):
            continue
        article_url = urljoin(page_url, anchor["href"])
        if article_url in seen_page:
            continue
        seen_page.add(article_url)
        page_links.append((article_url, title))
    return page_number, page_links, insecure_tls


def _iter_nbs_archive_links(
    session: requests.Session,
    *,
    max_pages: int = 36,
    max_empty_pages: int = 4,
) -> tuple[list[tuple[str, str]], bool]:
    del session
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    insecure_used = False
    empty_pages = 0
    worker_count = min(NBS_MAX_WORKERS, max_pages)
    page_results: list[tuple[int, list[tuple[str, str]], bool]] = []
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_fetch_nbs_archive_page, page_number): page_number
            for page_number in range(1, max_pages + 1)
        }
        for future in as_completed(future_map):
            page_number = future_map[future]
            try:
                page_results.append(future.result())
            except Exception:
                page_results.append((page_number, [], False))

    for _, page_links, insecure_tls in sorted(page_results, key=lambda item: item[0]):
        insecure_used = insecure_used or insecure_tls
        page_matches = 0
        for article_url, title in page_links:
            if article_url in seen:
                continue
            seen.add(article_url)
            links.append((article_url, title))
            page_matches += 1
        if page_matches == 0 and links:
            empty_pages += 1
            if empty_pages >= max_empty_pages:
                break
        else:
            empty_pages = 0
    return links, insecure_used


def _extract_nbs_search_bootstrap(html: str) -> dict[str, str]:
    match = re.search(
        r"initPubProperty\(\s*'(?P<site_code>[^']+)'\s*,\s*'(?P<tab>[^']+)'\s*,\s*'(?P<qt>[^']*)'\s*,\s*"
        r"'(?P<debug>[^']*)'\s*,\s*'(?P<page>[^']+)'\s*,\s*'(?P<page_size>[^']+)'\s*,\s*"
        r"'(?P<timestamp>[^']+)'\s*,\s*'(?P<word_token>[^']+)'\s*,\s*'(?P<tab_token>[^']+)'\s*,\s*"
        r"attrs,\s*'(?P<api_url>[^']+)'.*?'(?P<suid>[^']+)'\s*\)",
        html,
        re.S,
    )
    if not match:
        raise ValueError("NBS search bootstrap not found.")
    return {
        "siteCode": match.group("site_code"),
        "tab": match.group("tab"),
        "page": match.group("page"),
        "pageSize": match.group("page_size"),
        "timestamp": match.group("timestamp"),
        "wordToken": match.group("word_token"),
        "tabToken": match.group("tab_token"),
        "apiUrl": match.group("api_url"),
        "suid": match.group("suid"),
    }


def _iter_nbs_search_results(
    session: requests.Session,
    *,
    queries: tuple[str, ...] = NBS_SEARCH_QUERIES,
    page_size: int = 50,
    max_pages: int = 8,
) -> tuple[list[dict[str, Any]], bool]:
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    insecure_used = False
    stagnant_queries = 0

    for query in queries:
        before_query = len(results)
        search_url = f"{NBS_SEARCH_URL}?{urlencode({'qt': query})}"
        html, search_insecure = _request_text(session, search_url, timeout=90)
        insecure_used = insecure_used or search_insecure
        bootstrap = _extract_nbs_search_bootstrap(html)

        first_payload = {
            "siteCode": bootstrap["siteCode"],
            "tab": bootstrap["tab"],
            "timestamp": bootstrap["timestamp"],
            "wordToken": bootstrap["wordToken"],
            "page": 1,
            "pageSize": page_size,
            "qt": query,
        }
        response, api_insecure = _request(
            session,
            urljoin(bootstrap["apiUrl"], "s"),
            method="POST",
            data=first_payload,
            headers={"suid": bootstrap["suid"]},
            timeout=90,
        )
        insecure_used = insecure_used or api_insecure
        payload = response.json()
        total_hits = int(payload.get("data", {}).get("search", {}).get("totalHits", 0))
        total_pages = min(max_pages, max(1, int(np.ceil(total_hits / page_size)) if total_hits else 1))

        def collect_items(items: list[dict[str, Any]]) -> None:
            for item in items:
                title = str(item.get("title", "")).strip()
                article_url = str(item.get("viewUrl", "")).strip().replace("http://", "https://")
                if not title or not article_url:
                    continue
                if "/english/PressRelease/" not in article_url:
                    continue
                if not _is_nbs_ppi_title(title):
                    continue
                if article_url in seen:
                    continue
                seen.add(article_url)
                quick_description = str(item.get("myValues", {}).get("QUICKDESCRIPTION", "")).strip()
                summary = str(item.get("summary", "")).strip()
                doc_date = str(item.get("docDate", "")).strip()
                results.append(
                    {
                        "url": article_url,
                        "title": title,
                        "quick_description": quick_description,
                        "summary": summary,
                        "doc_date": doc_date,
                    }
                )

        collect_items(payload.get("data", {}).get("search", {}).get("searchs", []))

        for page_number in range(2, total_pages + 1):
            page_payload = dict(first_payload)
            page_payload["page"] = page_number
            response, page_insecure = _request(
                session,
                urljoin(bootstrap["apiUrl"], "s"),
                method="POST",
                data=page_payload,
                headers={"suid": bootstrap["suid"]},
                timeout=90,
            )
            insecure_used = insecure_used or page_insecure
            page_data = response.json()
            collect_items(page_data.get("data", {}).get("search", {}).get("searchs", []))

        if len(results) == before_query and results:
            stagnant_queries += 1
            if stagnant_queries >= 2:
                break
        else:
            stagnant_queries = 0

    return results, insecure_used


def _parse_nbs_search_result_item(item: dict[str, Any]) -> dict[str, Any]:
    title = str(item.get("title", "")).strip()
    observation_date = _parse_observation_month(re.sub(r"^\d+\.\s*", "", title))
    release_date = _coerce_datetime(pd.Series([item.get("doc_date")])).iloc[0]
    text_candidates = [
        _normalize_nbs_metric_text(str(item.get("quick_description", ""))),
        _normalize_nbs_metric_text(str(item.get("summary", ""))),
    ]
    for candidate in text_candidates:
        if not candidate:
            continue
        yoy = _try_parse_nbs_paragraph_metric(candidate, "yoy")
        if yoy is None:
            continue
        return {
            "date": observation_date,
            "value": yoy,
            "release_date": release_date,
        }
    raise ValueError("Could not parse NBS search result snippet.")


def _fetch_nbs_yoy_row(article_url: str, title: str) -> tuple[dict[str, Any], bool]:
    session = _new_session()
    html, article_insecure = _request_text(session, article_url, timeout=60)
    yoy_row, _ = _parse_nbs_article_html(
        html,
        article_url=article_url,
        title_hint=title,
    )
    return yoy_row, article_insecure


def _failed_source_result(
    key: str,
    label: str,
    source_url: str,
    detail: str,
    *,
    insecure_tls: bool = False,
    metadata: dict[str, Any] | None = None,
) -> SourceResult:
    return SourceResult(
        key=key,
        label=label,
        frame=_empty_monthly_frame(),
        source_url=source_url,
        ok=False,
        detail=detail,
        latest_observation=None,
        fetched_at=_utc_now(),
        freshness_days=_freshness_days_for_key(key),
        stale=False,
        insecure_tls=insecure_tls,
        metadata=metadata or {},
    )


def _serialize_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _deserialize_timestamp(value: Any) -> pd.Timestamp | None:
    if value in (None, "", pd.NaT):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    parsed = pd.Timestamp(parsed)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert(None)
    return parsed


def _source_result_to_payload(source: SourceResult) -> dict[str, Any]:
    return {
        "key": source.key,
        "label": source.label,
        "frame": source.frame.copy(deep=True),
        "source_url": source.source_url,
        "ok": source.ok,
        "detail": source.detail,
        "latest_observation": _serialize_timestamp(source.latest_observation),
        "fetched_at": _serialize_timestamp(source.fetched_at),
        "freshness_days": source.freshness_days,
        "stale": source.stale,
        "insecure_tls": source.insecure_tls,
        "metadata": dict(source.metadata),
    }


def _source_result_from_payload(payload: dict[str, Any]) -> SourceResult:
    frame = payload.get("frame", _empty_monthly_frame())
    if not isinstance(frame, pd.DataFrame):
        frame = pd.DataFrame(frame)
    return SourceResult(
        key=str(payload.get("key", "")),
        label=str(payload.get("label", "")),
        frame=frame.copy(deep=True),
        source_url=str(payload.get("source_url", "")),
        ok=bool(payload.get("ok", False)),
        detail=str(payload.get("detail", "")),
        latest_observation=_deserialize_timestamp(payload.get("latest_observation")),
        fetched_at=_deserialize_timestamp(payload.get("fetched_at")),
        freshness_days=payload.get("freshness_days"),
        stale=bool(payload.get("stale", False)),
        insecure_tls=bool(payload.get("insecure_tls", False)),
        metadata=dict(payload.get("metadata", {})),
    )


def _write_snapshot(path: Path, payloads: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "saved_at": _serialize_timestamp(_utc_now()),
        "payloads": payloads,
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(path)


def _read_snapshot(
    path: Path,
    *,
    max_age_seconds: int,
    allowed_keys: set[str] | frozenset[str] | None = None,
) -> dict[str, dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception:
        return None

    if not isinstance(bundle, dict):
        return None
    saved_at = _deserialize_timestamp(bundle.get("saved_at"))
    payloads = bundle.get("payloads")
    if saved_at is None or not isinstance(payloads, dict):
        return None
    age_seconds = (_utc_now() - pd.Timestamp(saved_at)).total_seconds()
    if age_seconds > max_age_seconds:
        return None
    if allowed_keys is not None:
        payloads = {key: value for key, value in payloads.items() if key in allowed_keys}
        if not payloads or not set(allowed_keys).issubset(payloads.keys()):
            return None
    return payloads


def _delete_snapshot(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        pass


def _restore_sources(payloads: dict[str, dict[str, Any]]) -> dict[str, SourceResult]:
    return {
        key: _source_result_from_payload(payload)
        for key, payload in payloads.items()
    }


def fetch_china_ppi_yoy(session: requests.Session) -> SourceResult:
    try:
        search_results, search_insecure = _iter_nbs_search_results(session)
    except Exception:
        search_results, search_insecure = [], False
    archive_links, archive_insecure = _iter_nbs_archive_links(session, max_pages=25)
    insecure_tls = search_insecure or archive_insecure
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    parsed_urls: set[str] = set()

    for item in search_results:
        try:
            rows.append(_parse_nbs_search_result_item(item))
            parsed_urls.add(str(item.get("url", "")))
        except Exception:
            parse_errors += 1

    combined_links: list[tuple[str, str]] = []
    seen: set[str] = set(parsed_urls)
    for article_url, title in archive_links:
        if article_url in seen:
            continue
        seen.add(article_url)
        combined_links.append((article_url, title))

    for item in search_results:
        article_url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip()
        if not article_url or article_url in parsed_urls or article_url in seen:
            continue
        seen.add(article_url)
        combined_links.append((article_url, title))

    if combined_links:
        worker_count = min(NBS_MAX_WORKERS, len(combined_links))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_fetch_nbs_yoy_row, article_url, title): (article_url, title)
                for article_url, title in combined_links
            }
            for future in as_completed(futures):
                try:
                    yoy_row, article_insecure = future.result()
                    insecure_tls = insecure_tls or article_insecure
                    rows.append(yoy_row)
                except Exception:
                    parse_errors += 1

    frame = _normalize_monthly_frame(pd.DataFrame(rows))
    detail = (
        f"parsed {len(frame)} monthly observations from {len(combined_links)} official links"
        if not frame.empty
        else f"no parseable NBS articles (errors={parse_errors})"
    )
    return _finalize_source_result(
        "china_ppi_yoy",
        "China PPI YoY",
        frame,
        NBS_SEARCH_URL,
        insecure_tls=insecure_tls,
        detail=detail,
        metadata={
            "search_results": len(search_results),
            "search_snippet_rows": len(parsed_urls),
            "archive_links": len(archive_links),
            "combined_links": len(combined_links),
            "parse_errors": parse_errors,
        },
    )


def _parse_ism_prices_from_text(text: str) -> float | None:
    normalized = re.sub(r"\s+", " ", text)
    patterns = (
        re.compile(
            r"Prices Index(?:[^.]{0,120})?registered\s+(\d{1,3}(?:\.\d+)?)\s+percent",
            re.IGNORECASE,
        ),
        re.compile(
            r"Prices Index(?:[^.]{0,120}?)(?:was|at|to)\s+(\d{1,3}(?:\.\d+)?)\s+percent",
            re.IGNORECASE,
        ),
        re.compile(
            r"prices paid(?:[^.]{0,120}?)(\d{1,3}(?:\.\d+)?)\s+percent",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        value = float(match.group(1))
        if 0.0 <= value <= 100.0:
            return value
    return None


def _parse_ism_observation_from_url(url: str) -> pd.Timestamp | None:
    match = ISM_URL_PATTERN.search(url)
    if not match:
        title_match = re.search(
            r"/(?P<year>\d{4})/(?P<month>[a-z]+)/?$",
            url,
            re.IGNORECASE,
        )
        if title_match:
            year = int(title_match.group("year"))
            month_name = title_match.group("month").lower()
            if month_name in MONTH_TO_NUMBER:
                return pd.Timestamp(year=year, month=MONTH_TO_NUMBER[month_name], day=1)
        return None

    month_name = (match.group("month") or "").lower()
    if month_name not in MONTH_TO_NUMBER:
        return None
    year = match.group("year")
    if year is None:
        year = str(_utc_now().year)
    return pd.Timestamp(year=int(year), month=MONTH_TO_NUMBER[month_name], day=1)


def _looks_like_login_page(text: str, *, final_url: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered or token in final_url.lower()
        for token in (
            "sign in",
            "login",
            "single sign on",
            "/login",
            "captcha_form",
            "grecaptcha",
            "captcha_resp",
            "recaptcha",
            "ecommerce.ismworld.org",
        )
    )


def _fetch_ism_direct_pages(
    session: requests.Session,
    *,
    months_back: int = 12,
) -> tuple[pd.DataFrame, bool, int, bool]:
    rows: list[dict[str, Any]] = []
    insecure_tls = False
    parse_failures = 0
    seen_urls: set[str] = set()
    blocked = False

    periods = pd.period_range(end=_utc_now().to_period("M"), periods=months_back, freq="M")
    for period in periods[::-1]:
        month_slug = period.strftime("%B").lower()
        url = urljoin(ISM_DIRECT_REPORT_ROOT, f"{month_slug}/")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        try:
            response, page_insecure = _request(session, url)
        except Exception:
            parse_failures += 1
            continue

        insecure_tls = insecure_tls or page_insecure
        text = response.text
        if _looks_like_login_page(text, final_url=str(response.url)):
            parse_failures += 1
            blocked = True
            break

        value = _parse_ism_prices_from_text(text)
        if value is None:
            parse_failures += 1
            continue

        rows.append(
            {
                "date": period.to_timestamp(),
                "value": value,
                "release_date": pd.Timestamp.today().normalize(),
            }
        )

    return _normalize_monthly_frame(pd.DataFrame(rows)), insecure_tls, parse_failures, blocked


def _extract_pdf_links_from_html(html: str, *, page_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = urljoin(page_url, anchor["href"])
        if not href.lower().endswith(".pdf"):
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append(href)
    return links


def _fetch_ism_roundup_pages(
    session: requests.Session,
    *,
    max_urls: int = 96,
) -> tuple[pd.DataFrame, bool, int, bool]:
    xml_text, insecure_tls = _request_text(session, ISM_SITEMAP_URL, timeout=90)
    urls = re.findall(r"<loc>([^<]+)</loc>", xml_text)
    filtered_urls = [
        url
        for url in urls
        if "report-on-business-roundup" in url.lower()
        and "manufacturing-pmi" in url.lower()
    ]

    rows: list[dict[str, Any]] = []
    parse_failures = 0
    blocked = False
    for page_url in filtered_urls[:max_urls]:
        observation = _parse_ism_observation_from_url(page_url)
        if observation is None:
            parse_failures += 1
            continue
        try:
            response, page_insecure = _request(session, page_url)
        except Exception:
            parse_failures += 1
            continue

        insecure_tls = insecure_tls or page_insecure
        html = response.text
        if _looks_like_login_page(html, final_url=str(response.url)):
            parse_failures += 1
            blocked = True
            break

        parsed = _parse_ism_prices_from_text(html)
        if parsed is None:
            pdf_urls = _extract_pdf_links_from_html(html, page_url=page_url)
            for pdf_url in pdf_urls:
                try:
                    pdf_bytes, pdf_insecure = _request_bytes(session, pdf_url, timeout=90)
                except Exception:
                    continue
                insecure_tls = insecure_tls or pdf_insecure
                parsed = _parse_ism_prices_from_text(pdf_bytes.decode("latin1", errors="ignore"))
                if parsed is not None:
                    break

        if parsed is None:
            parse_failures += 1
            continue

        rows.append(
            {
                "date": observation,
                "value": parsed,
                "release_date": observation,
            }
        )

    frame = _normalize_monthly_frame(pd.DataFrame(rows))
    return frame, insecure_tls, parse_failures, blocked


def fetch_ism_prices_paid(session: requests.Session) -> SourceResult:
    direct_frame, direct_insecure, direct_failures, direct_blocked = _fetch_ism_direct_pages(session)
    try:
        roundup_frame, roundup_insecure, roundup_failures, roundup_blocked = _fetch_ism_roundup_pages(session)
    except Exception:
        roundup_frame, roundup_insecure, roundup_failures, roundup_blocked = _empty_monthly_frame(), False, 0, False

    combined = pd.concat([direct_frame, roundup_frame], ignore_index=True)
    frame = _normalize_monthly_frame(combined)
    detail = (
        "official ISM pages returned a captcha/login wall; "
        if frame.empty and (direct_blocked or roundup_blocked)
        else ""
    )
    detail += (
        f"roundup_rows={len(roundup_frame)}, direct_rows={len(direct_frame)}, "
        f"roundup_failures={roundup_failures}, direct_failures={direct_failures}"
    )
    return _finalize_source_result(
        "ism_prices_paid",
        "ISM Prices Paid",
        frame,
        ISM_SITEMAP_URL,
        insecure_tls=direct_insecure or roundup_insecure,
        detail=detail,
        metadata={
            "direct_rows": len(direct_frame),
            "roundup_rows": len(roundup_frame),
            "direct_failures": direct_failures,
            "roundup_failures": roundup_failures,
            "direct_blocked": direct_blocked,
            "roundup_blocked": roundup_blocked,
        },
    )


def _parse_cleveland_nowcast_payload(payload: Any) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    charts = payload if isinstance(payload, list) else [payload]
    if not charts:
        return pd.DataFrame(), None

    chart = charts[-1]
    labels = [item.get("label", "") for item in chart.get("categories", [{}])[0].get("category", [])]
    datasets = chart.get("dataset", [])
    mapping = {
        "CPI Inflation": "cpi",
        "Core CPI Inflation": "core_cpi",
        "PCE Inflation": "pce",
        "Core PCE Inflation": "core_pce",
        "Actual CPI Inflation": "actual_cpi",
        "Actual Core CPI Inflation": "actual_core_cpi",
        "Actual PCE Inflation": "actual_pce",
        "Actual Core PCE Inflation": "actual_core_pce",
    }

    frame = pd.DataFrame({"label": labels, "step": np.arange(len(labels), dtype=int)})
    for dataset in datasets:
        series_name = dataset.get("seriesname", "")
        column = mapping.get(series_name)
        if column is None:
            continue
        values = [point.get("value") for point in dataset.get("data", [])]
        if len(values) < len(frame):
            values.extend([np.nan] * (len(frame) - len(values)))
        frame[column] = pd.to_numeric(_clean_series_values(pd.Series(values[: len(frame)])), errors="coerce")

    update_comment = chart.get("chart", {}).get("_comment")
    update_timestamp = pd.to_datetime(update_comment, errors="coerce")
    if pd.isna(update_timestamp):
        update_timestamp = None
    return frame, update_timestamp


def fetch_cleveland_nowcast(session: requests.Session) -> dict[str, SourceResult]:
    results: dict[str, SourceResult] = {}
    payloads = {
        "cleveland_nowcast_month": (CLEVELAND_MONTH_JSON_URL, "Cleveland Fed Inflation Nowcast (Monthly)"),
        "cleveland_nowcast_quarter": (
            CLEVELAND_QUARTER_JSON_URL,
            "Cleveland Fed Inflation Nowcast (Quarterly)",
        ),
    }
    for key, (url, label) in payloads.items():
        try:
            text, insecure_tls = _request_text(session, url, timeout=90)
            payload = json.loads(text)
            frame, latest = _parse_cleveland_nowcast_payload(payload)
            detail = f"{len(frame)} update steps in current chart cycle"
            results[key] = _finalize_source_result(
                key,
                label,
                frame,
                url,
                insecure_tls=insecure_tls,
                detail=detail,
                latest_override=latest,
            )
        except Exception as exc:
            results[key] = _failed_source_result(key, label, url, str(exc))
    return results


def _fixture_monthly_level(
    dates: pd.DatetimeIndex,
    signal: pd.Series,
    *,
    base: float = 100.0,
    base_rate: float = 0.002,
    scale: float = 0.001,
) -> pd.DataFrame:
    series_signal = pd.Series(signal, index=dates).bfill().ffill()
    monthly_rate = base_rate + scale * series_signal
    values = base * np.exp(np.cumsum(monthly_rate.to_numpy()))
    return _normalize_monthly_frame(pd.DataFrame({"date": dates, "value": values}))


def _fixture_mode() -> bool:
    return os.environ.get("INFLATION_MONITOR_FIXTURE_MODE", "").strip().lower() in {"1", "true", "yes"}


def _build_fixture_sources() -> dict[str, SourceResult]:
    dates = pd.date_range(end=_utc_now().to_period("M").to_timestamp(), periods=168, freq="MS")
    base_index = np.linspace(0.0, 14.0, len(dates))

    signal_a = pd.Series(np.sin(base_index) + 0.15 * np.cos(base_index / 2.0), index=dates)
    signal_b = pd.Series(np.cos(base_index / 1.5) + 0.08 * np.linspace(-1.0, 1.0, len(dates)), index=dates)
    signal_c = pd.Series(np.sin(base_index / 1.9) + 0.1 * np.linspace(0.0, 1.0, len(dates)), index=dates)
    signal_d = pd.Series(np.cos(base_index / 2.5) + 0.1 * np.sin(base_index / 4.0), index=dates)
    signal_e = pd.Series(np.sin(base_index / 1.35) + 0.18 * np.cos(base_index / 3.2), index=dates)
    signal_f = pd.Series(np.cos(base_index / 1.15) + 0.12 * np.sin(base_index / 2.8), index=dates)

    levels = {
        "used_vehicle_ppi": _fixture_monthly_level(dates, signal_a, base=100.0, base_rate=0.0026, scale=0.0014),
        "used_car_cpi": _fixture_monthly_level(
            dates,
            signal_a.shift(4).bfill(),
            base=100.0,
            base_rate=0.0021,
            scale=0.0012,
        ),
        "fao_food_price_index": _fixture_monthly_level(
            dates,
            signal_b,
            base=96.0,
            base_rate=0.0023,
            scale=0.0013,
        ),
        "food_home_cpi": _fixture_monthly_level(
            dates,
            signal_b.shift(3).bfill(),
            base=100.0,
            base_rate=0.0020,
            scale=0.0011,
        ),
        "wti_crude_oil": _fixture_monthly_level(
            dates,
            signal_e,
            base=72.0,
            base_rate=0.0018,
            scale=0.0030,
        ),
        "motor_fuel_cpi": _fixture_monthly_level(
            dates,
            signal_e.shift(2).bfill(),
            base=100.0,
            base_rate=0.0019,
            scale=0.0021,
        ),
        "henry_hub_gas": _fixture_monthly_level(
            dates,
            signal_f,
            base=4.0,
            base_rate=0.0022,
            scale=0.0028,
        ),
        "utility_gas_cpi": _fixture_monthly_level(
            dates,
            signal_f.shift(1).bfill(),
            base=100.0,
            base_rate=0.0018,
            scale=0.0018,
        ),
        "apparel_cpi": _fixture_monthly_level(
            dates,
            signal_c.shift(2).bfill(),
            base=100.0,
            base_rate=0.0017,
            scale=0.0011,
        ),
        "case_shiller": _fixture_monthly_level(
            dates,
            signal_c,
            base=120.0,
            base_rate=0.0030,
            scale=0.0014,
        ),
        "shelter_cpi": _fixture_monthly_level(
            dates,
            signal_c.shift(12).bfill(),
            base=100.0,
            base_rate=0.0022,
            scale=0.0008,
        ),
        "headline_ppi": _fixture_monthly_level(
            dates,
            signal_d,
            base=100.0,
            base_rate=0.0025,
            scale=0.0015,
        ),
        "headline_cpi": _fixture_monthly_level(
            dates,
            signal_d.shift(4).bfill(),
            base=100.0,
            base_rate=0.0021,
            scale=0.0010,
        ),
        "core_ppi": _fixture_monthly_level(
            dates,
            signal_d * 0.8 + signal_b * 0.2,
            base=100.0,
            base_rate=0.0024,
            scale=0.0012,
        ),
        "core_cpi": _fixture_monthly_level(
            dates,
            (signal_d * 0.8 + signal_b * 0.2).shift(5).bfill(),
            base=100.0,
            base_rate=0.0020,
            scale=0.0009,
        ),
        "china_import_price": _fixture_monthly_level(
            dates,
            signal_c.shift(3).bfill(),
            base=100.0,
            base_rate=0.0017,
            scale=0.0010,
        ),
    }

    china_ppi_yoy = _normalize_monthly_frame(
        pd.DataFrame(
            {
                "date": dates,
                "value": 0.4 + 2.2 * signal_c.to_numpy(),
            }
        )
    )
    ism_prices = _normalize_monthly_frame(
        pd.DataFrame(
            {
                "date": dates,
                "value": 55.0 + 10.0 * signal_d.to_numpy(),
            }
        )
    )
    sticky_cpi = _normalize_monthly_frame(
        pd.DataFrame(
            {
                "date": dates,
                "value": 2.2 + 0.9 * signal_c.shift(2).bfill().to_numpy(),
            }
        )
    )
    wage_growth = _normalize_monthly_frame(
        pd.DataFrame(
            {
                "date": dates,
                "value": 3.8 + 0.6 * signal_b.shift(1).bfill().to_numpy(),
            }
        )
    )

    month_labels = ["04/01", "04/02", "04/03", "PCE Feb", "04/10", "CPI Mar", "04/13"]
    month_frame = pd.DataFrame(
        {
            "label": month_labels,
            "step": np.arange(len(month_labels), dtype=int),
            "cpi": [0.31, 0.32, 0.32, 0.33, 0.34, 0.35, 0.35],
            "core_cpi": [0.28, 0.28, 0.27, 0.27, 0.26, 0.26, 0.26],
            "pce": [0.22, 0.23, 0.24, 0.25, 0.25, 0.25, 0.25],
            "core_pce": [0.24, 0.24, 0.24, 0.25, 0.25, 0.25, 0.25],
            "actual_cpi": [np.nan, np.nan, np.nan, np.nan, np.nan, 0.35, 0.35],
            "actual_core_cpi": [np.nan, np.nan, np.nan, np.nan, np.nan, 0.26, 0.26],
            "actual_pce": [np.nan, np.nan, np.nan, 0.25, 0.25, 0.25, 0.25],
            "actual_core_pce": [np.nan, np.nan, np.nan, 0.25, 0.25, 0.25, 0.25],
        }
    )
    quarter_frame = pd.DataFrame(
        {
            "label": ["Q1", "Q2", "Q3", "Q4"],
            "step": [0, 1, 2, 3],
            "cpi": [2.8, 2.9, 3.0, 3.1],
            "core_cpi": [3.0, 3.0, 3.1, 3.1],
        }
    )

    sources: dict[str, SourceResult] = {}
    for key, (series_id, label) in FRED_SERIES.items():
        frame = levels.get(key, _empty_monthly_frame())
        sources[key] = _finalize_source_result(
            key,
            label,
            frame,
            f"https://fred.stlouisfed.org/series/{series_id}",
        )

    extra_monthly = {
        "fao_food_price_index": ("FAO Food Price Index", FAO_PAGE_URL, levels["fao_food_price_index"]),
        "china_ppi_yoy": ("China PPI YoY", NBS_ARCHIVE_URL, china_ppi_yoy),
        "atlanta_sticky_cpi": ("Atlanta Fed Sticky CPI", ATLANTA_STICKY_CPI_URL, sticky_cpi),
        "atlanta_wage_growth": ("Atlanta Fed Wage Growth Tracker", ATLANTA_WAGE_GROWTH_URL, wage_growth),
        "ism_prices_paid": ("ISM Prices Paid", ISM_SITEMAP_URL, ism_prices),
    }
    for key, (label, url, frame) in extra_monthly.items():
        sources[key] = _finalize_source_result(key, label, frame, url)

    latest_nowcast = pd.Timestamp("2026-04-13")
    sources["cleveland_nowcast_month"] = _finalize_source_result(
        "cleveland_nowcast_month",
        "Cleveland Fed Inflation Nowcast (Monthly)",
        month_frame,
        CLEVELAND_MONTH_JSON_URL,
        latest_override=latest_nowcast,
    )
    sources["cleveland_nowcast_quarter"] = _finalize_source_result(
        "cleveland_nowcast_quarter",
        "Cleveland Fed Inflation Nowcast (Quarterly)",
        quarter_frame,
        CLEVELAND_QUARTER_JSON_URL,
        latest_override=latest_nowcast,
    )
    return sources


@st.cache_data(ttl=MONTHLY_CACHE_TTL_SECONDS, show_spinner=False)
def load_monthly_sources(_refresh_token: int = 0, force_live: bool = False) -> dict[str, dict[str, Any]]:
    del _refresh_token
    if _fixture_mode():
        fixture_sources = _build_fixture_sources()
        return {
            key: _source_result_to_payload(value)
            for key, value in fixture_sources.items()
            if not key.startswith("cleveland_nowcast_")
        }

    if not force_live:
        snapshot = _read_snapshot(
            MONTHLY_SNAPSHOT_PATH,
            max_age_seconds=MONTHLY_SNAPSHOT_MAX_AGE_SECONDS,
            allowed_keys=MONTHLY_SOURCE_KEYS,
        )
        if snapshot is not None:
            return snapshot

    session = _new_session()
    results: dict[str, SourceResult] = {}

    fred_worker_count = min(FRED_MAX_WORKERS, len(FRED_SERIES))
    with ThreadPoolExecutor(max_workers=fred_worker_count) as executor:
        future_map = {
            executor.submit(
                fetch_fred_series, _new_session(), key=key, series_id=series_id, label=label
            ): (key, series_id, label)
            for key, (series_id, label) in FRED_SERIES.items()
        }
        for future in as_completed(future_map):
            key, series_id, label = future_map[future]
            try:
                results[key] = future.result()
            except Exception as exc:
                results[key] = _failed_source_result(
                    key,
                    label,
                    f"https://fred.stlouisfed.org/series/{series_id}",
                    str(exc),
                )

    loaders = (
        ("fao_food_price_index", "FAO Food Price Index", FAO_PAGE_URL, fetch_fao_food_price_index),
        ("atlanta_wage_growth", "Atlanta Fed Wage Growth Tracker", ATLANTA_WAGE_GROWTH_URL, fetch_atlanta_wage_growth),
        ("atlanta_sticky_cpi", "Atlanta Fed Sticky CPI", ATLANTA_STICKY_CPI_URL, fetch_atlanta_sticky_cpi),
        ("china_ppi_yoy", "China PPI YoY", NBS_ARCHIVE_URL, fetch_china_ppi_yoy),
        ("ism_prices_paid", "ISM Prices Paid", ISM_SITEMAP_URL, fetch_ism_prices_paid),
    )
    for key, label, url, loader in loaders:
        try:
            results[key] = loader(session)
        except Exception as exc:
            results[key] = _failed_source_result(key, label, url, str(exc))

    payloads = {
        key: _source_result_to_payload(value)
        for key, value in results.items()
    }
    _write_snapshot(MONTHLY_SNAPSHOT_PATH, payloads)
    return payloads


@st.cache_data(ttl=CLEVELAND_CACHE_TTL_SECONDS, show_spinner=False)
def load_cleveland_sources(_refresh_token: int = 0, force_live: bool = False) -> dict[str, dict[str, Any]]:
    del _refresh_token
    if _fixture_mode():
        fixture_sources = _build_fixture_sources()
        return {
            key: _source_result_to_payload(value)
            for key, value in fixture_sources.items()
            if key.startswith("cleveland_nowcast_")
        }

    if not force_live:
        snapshot = _read_snapshot(
            CLEVELAND_SNAPSHOT_PATH,
            max_age_seconds=CLEVELAND_SNAPSHOT_MAX_AGE_SECONDS,
            allowed_keys=CLEVELAND_SOURCE_KEYS,
        )
        if snapshot is not None:
            return snapshot

    session = _new_session()
    payloads = {
        key: _source_result_to_payload(value)
        for key, value in fetch_cleveland_nowcast(session).items()
    }
    _write_snapshot(CLEVELAND_SNAPSHOT_PATH, payloads)
    return payloads


def load_all_sources(refresh_token: int = 0, force_live: bool = False) -> dict[str, SourceResult]:
    sources = _restore_sources(load_monthly_sources(refresh_token, force_live))
    sources.update(_restore_sources(load_cleveland_sources(refresh_token, force_live)))
    return sources


def _minimal_series(source: SourceResult) -> pd.Series:
    if source.frame.empty or "date" not in source.frame.columns:
        return pd.Series(dtype=float)
    series = pd.Series(source.frame["value"].to_numpy(), index=pd.to_datetime(source.frame["date"]))
    series = pd.to_numeric(series, errors="coerce").dropna()
    series = series[~series.index.duplicated(keep="last")]
    return series.sort_index()


def _transform_series(series: pd.Series, transform: str) -> pd.Series:
    base = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if transform == "raw":
        return base
    if transform == "yoy":
        return (base / base.shift(12) - 1.0) * 100.0
    raise ValueError(f"Unsupported transform: {transform}")


def _shift_forward(series: pd.Series, lag_months: int) -> pd.Series:
    if series.empty:
        return series
    shifted = series.copy()
    shifted.index = shifted.index + pd.DateOffset(months=int(lag_months))
    shifted = shifted[~shifted.index.duplicated(keep="last")]
    return shifted.sort_index()


def _joined(indicator_series: pd.Series, target_series: pd.Series) -> pd.DataFrame:
    frame = pd.concat(
        [
            pd.Series(indicator_series, name="indicator"),
            pd.Series(target_series, name="target"),
        ],
        axis=1,
        join="inner",
    )
    frame.index.name = "date"
    frame = frame.dropna().sort_index()
    if frame.empty:
        return pd.DataFrame(columns=["date", "indicator", "target"])
    return frame.reset_index()


def _series_to_frame(series: pd.Series, column: str) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["date", column])
    frame = pd.DataFrame({"date": pd.to_datetime(series.index), column: pd.to_numeric(series.to_numpy(), errors="coerce")})
    frame = frame.dropna(subset=["date", column]).sort_values("date").reset_index(drop=True)
    return frame


def _projection_tail_frame(
    shifted_indicator_series: pd.Series,
    target_series: pd.Series,
) -> pd.DataFrame:
    if shifted_indicator_series.empty:
        return pd.DataFrame(columns=["date", "indicator"])
    if target_series.empty:
        return _series_to_frame(shifted_indicator_series, "indicator")
    latest_target = pd.Timestamp(target_series.index.max())
    future = shifted_indicator_series.loc[shifted_indicator_series.index > latest_target]
    return _series_to_frame(future, "indicator")


def _projection_preview_frame(report: ValidationReport, months: int = 6) -> pd.DataFrame:
    if report.projection_frame.empty:
        return pd.DataFrame(columns=["month", "implied_value"])
    preview = report.projection_frame.copy().sort_values("date").head(months)
    preview["month"] = pd.to_datetime(preview["date"]).dt.strftime("%Y-%m")
    preview["implied_value"] = preview["indicator"].map(lambda value: _format_value(float(value)))
    return preview[["month", "implied_value"]]


def _source_freshness_reference(source: SourceResult) -> pd.Timestamp | None:
    reference = source.metadata.get("freshness_reference")
    parsed = _deserialize_timestamp(reference)
    if parsed is not None:
        return parsed
    if source.latest_observation is None:
        return None
    return pd.Timestamp(source.latest_observation) + pd.offsets.MonthEnd(1)


def _source_age_days(source: SourceResult) -> int | None:
    reference = _source_freshness_reference(source)
    if reference is None:
        return None
    return int((_utc_now().normalize() - pd.Timestamp(reference).normalize()).days)


def _latest_rolling_corr(frame: pd.DataFrame, window: int = 24) -> float | None:
    if frame.empty or len(frame) < max(3, window):
        return None
    corr = frame["indicator"].rolling(window).corr(frame["target"])
    corr = corr.dropna()
    if corr.empty:
        return None
    return float(corr.iloc[-1])


def _split_frame(frame: pd.DataFrame, ratio: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.copy()
    split_index = int(np.floor(len(frame) * ratio))
    split_index = min(max(split_index, 1), max(len(frame) - 1, 1))
    train = frame.iloc[:split_index].reset_index(drop=True)
    holdout = frame.iloc[split_index:].reset_index(drop=True)
    return train, holdout


def _choose_best_lag(
    indicator_series: pd.Series,
    target_series: pd.Series,
    *,
    lag_min: int,
    lag_max: int,
    train_ratio: float = 0.8,
) -> dict[str, Any]:
    best_train: dict[str, Any] | None = None
    best_holdout: dict[str, Any] | None = None
    lag_details: list[dict[str, Any]] = []

    for lag in range(int(lag_min), int(lag_max) + 1):
        aligned = _joined(_shift_forward(indicator_series, lag), target_series)
        if aligned.empty:
            continue
        train, holdout = _split_frame(aligned, ratio=train_ratio)
        train_corr = float(train["indicator"].corr(train["target"])) if len(train) >= 3 else None
        holdout_corr = float(holdout["indicator"].corr(holdout["target"])) if len(holdout) >= 3 else None
        full_corr = float(aligned["indicator"].corr(aligned["target"])) if len(aligned) >= 3 else None
        detail = {
            "lag": lag,
            "aligned": aligned,
            "train": train,
            "holdout": holdout,
            "train_corr": train_corr,
            "holdout_corr": holdout_corr,
            "full_corr": full_corr,
            "overlap": len(aligned),
        }
        lag_details.append(detail)

        if train_corr is not None:
            if best_train is None or (train_corr, -lag) > (best_train["train_corr"], -best_train["lag"]):
                best_train = detail
        if holdout_corr is not None:
            if best_holdout is None or (holdout_corr, -lag) > (best_holdout["holdout_corr"], -best_holdout["lag"]):
                best_holdout = detail

    return {
        "best_train": best_train,
        "best_holdout": best_holdout,
        "details": lag_details,
    }


def _fit_pass_through(frame: pd.DataFrame) -> tuple[float | None, float | None]:
    if frame.empty or len(frame) < 3:
        return None, None
    x = pd.to_numeric(frame["indicator"], errors="coerce")
    y = pd.to_numeric(frame["target"], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 3:
        return None, None
    variance = float(valid["x"].var(ddof=0))
    if np.isclose(variance, 0.0):
        return None, None
    covariance = float(((valid["x"] - valid["x"].mean()) * (valid["y"] - valid["y"].mean())).mean())
    beta = covariance / variance
    intercept = float(valid["y"].mean() - beta * valid["x"].mean())
    return float(beta), intercept


def validate_indicator(spec: IndicatorSpec, sources: dict[str, SourceResult]) -> ValidationReport:
    indicator_source = sources.get(spec.indicator_key)
    target_source = sources.get(spec.target_key)

    report = ValidationReport(
        key=spec.key,
        title=spec.title,
        indicator_key=spec.indicator_key,
        target_key=spec.target_key,
        indicator_label=DISPLAY_LABELS.get(spec.indicator_key, spec.indicator_key),
        target_label=DISPLAY_LABELS.get(spec.target_key, spec.target_key),
        approved=False,
        note=spec.note,
        secondary_axis=spec.secondary_axis,
        source_urls={
            spec.indicator_key: indicator_source.source_url if indicator_source else "",
            spec.target_key: target_source.source_url if target_source else "",
        },
    )

    if indicator_source is None or target_source is None:
        report.reasons.append("missing source definition")
        return report
    if not indicator_source.ok:
        report.reasons.append(f"indicator unavailable: {indicator_source.detail}")
        return report
    if not target_source.ok:
        report.reasons.append(f"target unavailable: {target_source.detail}")
        return report

    report.raw_indicator = indicator_source.frame.copy()
    report.target_frame = target_source.frame.copy()
    report.latest_indicator_date = indicator_source.latest_observation
    report.latest_target_date = target_source.latest_observation
    report.duplicate_months = bool(indicator_source.frame["date"].duplicated().any() or target_source.frame["date"].duplicated().any())
    report.freshness_ok = not indicator_source.stale and not target_source.stale
    source_age_days = [value for value in (_source_age_days(indicator_source), _source_age_days(target_source)) if value is not None]
    if source_age_days:
        report.stale_gap_days = max(source_age_days)
    if not report.freshness_ok and report.stale_gap_days is not None:
        report.freshness_note = f"stale by {report.stale_gap_days} days"

    indicator_series = _transform_series(_minimal_series(indicator_source), spec.indicator_transform)
    target_series = _transform_series(_minimal_series(target_source), spec.target_transform)
    indicator_series = indicator_series.dropna()
    target_series = target_series.dropna()
    if indicator_series.empty or target_series.empty:
        report.reasons.append("transformed series is empty")
        return report

    selection = _choose_best_lag(
        indicator_series,
        target_series,
        lag_min=spec.lag_min,
        lag_max=spec.lag_max,
    )
    best_train = selection["best_train"]
    best_holdout = selection["best_holdout"]
    if best_train is None:
        report.reasons.append("no valid lag produced overlap")
        return report

    report.train_best_lag = int(best_train["lag"])
    report.holdout_best_lag = int(best_holdout["lag"]) if best_holdout is not None else None
    report.lag_months = report.train_best_lag
    report.full_corr = float(best_train["full_corr"]) if best_train["full_corr"] is not None else None
    report.holdout_corr = float(best_train["holdout_corr"]) if best_train["holdout_corr"] is not None else None
    report.overlap = int(best_train["overlap"])
    report.aligned_frame = best_train["aligned"].copy()
    shifted_indicator = _shift_forward(indicator_series, report.lag_months)
    report.shifted_indicator_frame = _series_to_frame(shifted_indicator, "indicator")
    report.projection_frame = _projection_tail_frame(shifted_indicator, target_series)
    report.rolling_corr_24 = _latest_rolling_corr(report.aligned_frame, window=24)
    fit_frame = best_train["train"].copy() if isinstance(best_train.get("train"), pd.DataFrame) else report.aligned_frame.copy()
    beta, intercept = _fit_pass_through(fit_frame)
    report.pass_through_beta = beta
    report.pass_through_intercept = intercept
    fit_corr = best_train["train_corr"] if best_train.get("train_corr") is not None else report.full_corr
    if fit_corr is not None:
        report.score_weight = float(max(abs(float(fit_corr)), 0.05))

    if report.overlap < 48:
        report.reasons.append("overlap < 48 months")
    if report.duplicate_months:
        report.reasons.append("duplicate month detected")
    if not report.freshness_ok and (report.stale_gap_days is None or report.stale_gap_days > 92):
        report.reasons.append("freshness gap > 3 months")
    if report.holdout_best_lag is None:
        report.reasons.append("holdout lag unavailable")
    elif abs(report.train_best_lag - report.holdout_best_lag) > 3:
        report.reasons.append("train/holdout lag gap > 3 months")
    if report.full_corr is None or report.full_corr < 0.35:
        report.reasons.append("full-sample aligned corr < 0.35")
    if report.rolling_corr_24 is None or report.rolling_corr_24 <= 0:
        report.reasons.append("latest 24M rolling corr <= 0")

    report.approved = not report.reasons
    return report


def build_validation_reports(sources: dict[str, SourceResult]) -> list[ValidationReport]:
    return [validate_indicator(spec, sources) for spec in LEADING_SPECS]


def compute_forward_pressure(
    reports: list[ValidationReport],
) -> tuple[float | None, pd.DataFrame, str]:
    approved = [
        report
        for report in reports
        if report.approved
        and not report.shifted_indicator_frame.empty
        and report.score_weight is not None
    ]
    if len(approved) < 3:
        return None, pd.DataFrame(columns=["date", "score"]), "withheld: fewer than 3 approved leading pairs"

    z_frames: list[pd.DataFrame] = []
    for report in approved:
        shifted = report.shifted_indicator_frame.copy().sort_values("date")
        rolling_mean = shifted["indicator"].rolling(60, min_periods=24).mean()
        rolling_std = shifted["indicator"].rolling(60, min_periods=24).std()
        z = (shifted["indicator"] - rolling_mean) / rolling_std.replace(0, np.nan)
        z_frame = pd.DataFrame(
            {
                "date": shifted["date"],
                report.key: z,
                f"{report.key}__weighted": z * float(report.score_weight),
                f"{report.key}__weight": np.where(z.notna(), float(report.score_weight), np.nan),
            }
        )
        z_frames.append(z_frame)

    merged = z_frames[0]
    for frame in z_frames[1:]:
        merged = merged.merge(frame, on="date", how="outer")
    merged = merged.sort_values("date")
    weighted_columns = [column for column in merged.columns if column.endswith("__weighted")]
    weight_columns = [column for column in merged.columns if column.endswith("__weight")]
    score_columns = [column for column in merged.columns if column not in {"date", *weighted_columns, *weight_columns}]
    merged["weighted_sum"] = merged[weighted_columns].sum(axis=1, skipna=True)
    merged["weight_sum"] = merged[weight_columns].sum(axis=1, skipna=True)
    merged["score"] = merged["weighted_sum"] / merged["weight_sum"].replace(0, np.nan)
    merged = merged.dropna(subset=["score"]).reset_index(drop=True)
    if merged.empty:
        return None, pd.DataFrame(columns=["date", "score"]), "withheld: z-score history unavailable"

    latest_score = float(merged["score"].iloc[-1])
    for report in approved:
        if report.key in merged.columns:
            valid = merged[["date", report.key]].dropna()
            report.latest_zscore = float(valid[report.key].iloc[-1]) if not valid.empty else None
    horizon = _format_date(pd.to_datetime(merged["date"]).max()) if not merged.empty else "N/A"
    return latest_score, merged[["date", "score"]], f"{len(approved)} approved pairs | pass-through weighted | implied through {horizon}"


def build_source_status_frame(sources: dict[str, SourceResult]) -> pd.DataFrame:
    rows = []
    for source in sources.values():
        if not source.ok:
            status = "failed"
        elif source.stale:
            status = "stale"
        else:
            status = "ok"
        rows.append(
            {
                "source": source.label,
                "key": source.key,
                "status": status,
                "rows": source.row_count,
                "latest_observation": _format_date(source.latest_observation),
                "freshness_days": source.freshness_days,
                "stale": source.stale,
                "insecure_tls": source.insecure_tls,
                "detail": source.detail,
                "url": source.source_url,
            }
        )
    return pd.DataFrame(rows).sort_values(["status", "source"]).reset_index(drop=True)


def build_rejected_frame(reports: list[ValidationReport]) -> pd.DataFrame:
    rows = []
    for report in reports:
        if report.approved:
            continue
        rows.append(
            {
                "pair": report.title,
                "reasons": "; ".join(report.reasons) if report.reasons else "not approved",
                "lag_months": report.lag_months,
                "full_corr": None if report.full_corr is None else round(report.full_corr, 3),
                "rolling_corr_24": None if report.rolling_corr_24 is None else round(report.rolling_corr_24, 3),
                "overlap": report.overlap,
                "indicator_latest": _format_date(report.latest_indicator_date),
                "target_latest": _format_date(report.latest_target_date),
            }
        )
    return pd.DataFrame(rows)


def _watchlist_reports(reports: list[ValidationReport]) -> list[ValidationReport]:
    return [
        report
        for report in reports
        if report.key in WATCHLIST_KEYS
        and not report.approved
        and not report.aligned_frame.empty
    ]


def _format_value(value: float | None, *, suffix: str = "%", decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}{suffix}"


def _format_date(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return pd.Timestamp(value).strftime("%Y-%m")


def _latest_value(frame: pd.DataFrame, forecast_col: str, actual_col: str | None = None) -> float | None:
    if frame.empty or forecast_col not in frame.columns:
        return None

    forecast_idx = frame[forecast_col].last_valid_index()
    actual_idx = None
    if actual_col and actual_col in frame.columns:
        actual_idx = frame[actual_col].last_valid_index()

    if actual_idx is not None and (forecast_idx is None or actual_idx >= forecast_idx):
        value = frame.loc[actual_idx, actual_col]
        if pd.notna(value):
            return float(value)

    if forecast_idx is not None:
        value = frame.loc[forecast_idx, forecast_col]
        if pd.notna(value):
            return float(value)
    return None


def _series_with_actual_nowcast(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    replacements = {
        "cpi": "actual_cpi",
        "core_cpi": "actual_core_cpi",
        "pce": "actual_pce",
        "core_pce": "actual_core_pce",
    }
    for forecast_col, actual_col in replacements.items():
        if forecast_col not in data.columns:
            continue
        if actual_col not in data.columns:
            data[f"current_{forecast_col}"] = data[forecast_col]
            continue
        data[f"current_{forecast_col}"] = data[actual_col].combine_first(data[forecast_col])
    return data


def _current_nowcast_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Headline CPI", _latest_value(frame, "cpi", "actual_cpi")),
        ("Core CPI", _latest_value(frame, "core_cpi", "actual_core_cpi")),
        ("PCE", _latest_value(frame, "pce", "actual_pce")),
        ("Core PCE", _latest_value(frame, "core_pce", "actual_core_pce")),
    ]
    return pd.DataFrame(rows, columns=["series", "value"])


def _lookback_start(months: int = 60) -> pd.Timestamp:
    return (_utc_now().normalize() - pd.DateOffset(months=months)).replace(day=1)


def _filter_plot_range(frame: pd.DataFrame, *, months: int = 120, date_col: str = "date") -> pd.DataFrame:
    if frame.empty or date_col not in frame.columns:
        return frame.copy()
    start = _lookback_start(months)
    return frame.loc[pd.to_datetime(frame[date_col]) >= start].copy()


def build_leading_figure(report: ValidationReport, *, aligned_view: bool) -> go.Figure:
    if aligned_view:
        data = _filter_plot_range(report.aligned_frame, months=120)
        x_values = data["date"] if not data.empty else []
        indicator_values = data["indicator"] if not data.empty else []
        target_values = data["target"] if not data.empty else []
        projection = _filter_plot_range(report.projection_frame, months=120)
        figure = make_subplots(specs=[[{"secondary_y": report.secondary_axis}]])
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=indicator_values,
                mode="lines",
                name=f"{report.indicator_label} (aligned)",
                line={"color": "#1f5aa6", "width": 3},
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=target_values,
                mode="lines",
                name=f"{report.target_label}",
                line={"color": "#d4682d", "width": 3},
            ),
            secondary_y=report.secondary_axis,
        )
        if not projection.empty:
            projection_line = projection.copy()
            if not data.empty:
                projection_line = pd.concat(
                    [
                        data[["date", "indicator"]].tail(1),
                        projection[["date", "indicator"]],
                    ],
                    ignore_index=True,
                )
            figure.add_trace(
                go.Scatter(
                    x=projection_line["date"],
                    y=projection_line["indicator"],
                    mode="lines+markers",
                    name=f"{report.target_label} implied path",
                    line={"color": "#1f5aa6", "width": 3, "dash": "dash"},
                    marker={"size": 7},
                ),
                secondary_y=False,
            )
            if report.latest_target_date is not None:
                figure.add_vline(
                    x=pd.Timestamp(report.latest_target_date),
                    line_width=1,
                    line_dash="dot",
                    line_color="#94a3b8",
                )
        figure.update_layout(
            template="plotly_white",
            height=360,
            margin={"l": 12, "r": 12, "t": 36, "b": 12},
            legend={"orientation": "h", "y": 1.05},
            title=f"Aligned + Projection View, lag {report.lag_months}m",
        )
        return figure

    indicator = report.raw_indicator.rename(columns={"value": "indicator"})
    target = report.target_frame.rename(columns={"value": "target"})
    merged = indicator[["date", "indicator"]].merge(target[["date", "target"]], on="date", how="outer")
    merged = _filter_plot_range(merged.sort_values("date"), months=120)
    figure = make_subplots(specs=[[{"secondary_y": report.secondary_axis}]])
    figure.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["indicator"],
            mode="lines",
            name=report.indicator_label,
            line={"color": "#1f5aa6", "width": 3},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=merged["date"],
            y=merged["target"],
            mode="lines",
            name=report.target_label,
            line={"color": "#d4682d", "width": 3},
        ),
        secondary_y=report.secondary_axis,
    )
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 12, "r": 12, "t": 36, "b": 12},
        legend={"orientation": "h", "y": 1.05},
        title="Raw View",
    )
    return figure


def build_forward_pressure_figure(
    forward_frame: pd.DataFrame,
    *,
    actual_cutoff: pd.Timestamp | None,
) -> go.Figure:
    chart_frame = _filter_plot_range(forward_frame, months=60)
    figure = go.Figure()
    if chart_frame.empty:
        figure.update_layout(template="plotly_white", height=260, margin={"l": 12, "r": 12, "t": 12, "b": 12})
        return figure

    if actual_cutoff is not None:
        actual_cutoff = pd.Timestamp(actual_cutoff)
        historical = chart_frame.loc[pd.to_datetime(chart_frame["date"]) <= actual_cutoff].copy()
        projected = chart_frame.loc[pd.to_datetime(chart_frame["date"]) > actual_cutoff].copy()
    else:
        historical = chart_frame.copy()
        projected = pd.DataFrame(columns=chart_frame.columns)

    if not historical.empty:
        figure.add_trace(
            go.Scatter(
                x=historical["date"],
                y=historical["score"],
                mode="lines",
                line={"width": 3, "color": "#d4682d"},
                name="Forward Pressure Score",
            )
        )
    if not projected.empty:
        projected_line = projected.copy()
        if not historical.empty:
            projected_line = pd.concat(
                [
                    historical[["date", "score"]].tail(1),
                    projected[["date", "score"]],
                ],
                ignore_index=True,
            )
        figure.add_trace(
            go.Scatter(
                x=projected_line["date"],
                y=projected_line["score"],
                mode="lines+markers",
                line={"width": 3, "color": "#d4682d", "dash": "dash"},
                marker={"size": 7},
                name="Implied future score",
            )
        )
        if actual_cutoff is not None:
            figure.add_vline(x=actual_cutoff, line_width=1, line_dash="dot", line_color="#94a3b8")

    figure.update_layout(
        template="plotly_white",
        height=260,
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
        legend={"orientation": "h", "y": 1.05},
    )
    return figure


def build_nowcast_figure(month_source: SourceResult) -> go.Figure:
    data = _series_with_actual_nowcast(month_source.frame)
    figure = go.Figure()
    series_order = [
        ("current_cpi", "Headline CPI", "#b7410e"),
        ("current_core_cpi", "Core CPI", "#1f5aa6"),
        ("current_pce", "PCE", "#2d7f5e"),
        ("current_core_pce", "Core PCE", "#6e5aa6"),
    ]
    for column, label, color in series_order:
        if column not in data.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=data["label"],
                y=data[column],
                mode="lines+markers",
                name=label,
                line={"width": 3, "color": color},
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        legend={"orientation": "h", "y": 1.05},
        xaxis_title="Update step",
        yaxis_title="MoM nowcast (%)",
    )
    return figure


def build_persistence_figure(sticky_source: SourceResult, wage_source: SourceResult) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    sticky = _filter_plot_range(sticky_source.frame, months=72)
    wage = _filter_plot_range(wage_source.frame, months=72)
    if not sticky.empty:
        figure.add_trace(
            go.Scatter(
                x=sticky["date"],
                y=sticky["value"],
                mode="lines",
                name="Sticky CPI",
                line={"color": "#9f2f2f", "width": 3},
            ),
            secondary_y=False,
        )
    if not wage.empty:
        figure.add_trace(
            go.Scatter(
                x=wage["date"],
                y=wage["value"],
                mode="lines",
                name="Wage Growth",
                line={"color": "#235d55", "width": 3},
            ),
            secondary_y=True,
        )
    figure.update_layout(
        template="plotly_white",
        height=360,
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        legend={"orientation": "h", "y": 1.05},
    )
    figure.update_yaxes(title_text="Sticky CPI YoY (%)", secondary_y=False)
    figure.update_yaxes(title_text="Wage Growth (%)", secondary_y=True)
    return figure


def _health_counts(sources: dict[str, SourceResult]) -> dict[str, int]:
    ok = sum(1 for source in sources.values() if source.ok and not source.stale)
    stale = sum(1 for source in sources.values() if source.ok and source.stale)
    failed = sum(1 for source in sources.values() if not source.ok)
    insecure = sum(1 for source in sources.values() if source.insecure_tls)
    return {"ok": ok, "stale": stale, "failed": failed, "insecure": insecure}


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ink: #17324d;
            --muted: #64748b;
            --accent: #d4682d;
            --accent-soft: #f7ebe4;
            --panel: #f8fafc;
            --border: #d9e2ec;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.25rem 1.4rem;
            border: 1px solid var(--border);
            border-radius: 20px;
            background: linear-gradient(135deg, #fffaf6 0%, #f8fbff 100%);
            margin-bottom: 1rem;
        }
        .metric-card {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: white;
            padding: 0.95rem 1rem;
            min-height: 112px;
            box-shadow: 0 4px 14px rgba(23, 50, 77, 0.05);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.86rem;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.6rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-note {
            color: var(--muted);
            font-size: 0.82rem;
            margin-top: 0.35rem;
        }
        .section-chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.02em;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(
    sources: dict[str, SourceResult],
    reports: list[ValidationReport],
    forward_score: float | None,
    forward_frame: pd.DataFrame,
    forward_note: str,
) -> None:
    st.header("Overview")

    headline_cpi = _transform_series(_minimal_series(sources["headline_cpi"]), "yoy").dropna()
    core_cpi = _transform_series(_minimal_series(sources["core_cpi"]), "yoy").dropna()
    month_nowcast = sources["cleveland_nowcast_month"]
    nowcast_snapshot = _current_nowcast_snapshot(month_nowcast.frame)
    health = _health_counts(sources)

    metric_columns = st.columns(4)
    with metric_columns[0]:
        _metric_card(
            "Headline CPI YoY",
            _format_value(float(headline_cpi.iloc[-1]) if not headline_cpi.empty else None),
            f"Latest month: {_format_date(sources['headline_cpi'].latest_observation)}",
        )
    with metric_columns[1]:
        _metric_card(
            "Core CPI YoY",
            _format_value(float(core_cpi.iloc[-1]) if not core_cpi.empty else None),
            f"Latest month: {_format_date(sources['core_cpi'].latest_observation)}",
        )
    with metric_columns[2]:
        headline_nowcast = None
        if not nowcast_snapshot.empty:
            headline_nowcast = float(nowcast_snapshot.loc[nowcast_snapshot["series"] == "Headline CPI", "value"].iloc[0])
        _metric_card(
            "Cleveland Nowcast",
            _format_value(headline_nowcast),
            f"Updated: {_format_date(month_nowcast.latest_observation)}",
        )
    with metric_columns[3]:
        forward_value = "Withheld" if forward_score is None else _format_value(forward_score, suffix="", decimals=2)
        _metric_card(
            "Forward Pressure Score",
            forward_value,
            forward_note,
        )

    summary_columns = st.columns([1.6, 1.2])
    with summary_columns[0]:
        approved = [report for report in reports if report.approved]
        actual_cutoff = max(
            (report.latest_target_date for report in approved if report.latest_target_date is not None),
            default=None,
        )
        st.markdown('<span class="section-chip">Source Health</span>', unsafe_allow_html=True)
        st.caption(
            f"OK {health['ok']} | Stale {health['stale']} | Failed {health['failed']} | Insecure TLS fallback {health['insecure']}"
        )
        if forward_score is not None and not forward_frame.empty:
            figure = build_forward_pressure_figure(forward_frame, actual_cutoff=actual_cutoff)
            st.plotly_chart(figure, width="stretch", config=PLOTLY_CONFIG)
            st.caption("Forward Pressure Score는 승인된 선행지표의 historical pass-through 강도로 가중합니다. 아직 headline CPI basket weight까지 보정한 점수는 아닙니다.")
            if actual_cutoff is not None and pd.to_datetime(forward_frame["date"]).max() > pd.Timestamp(actual_cutoff):
                st.caption(
                    f"점선은 승인된 선행지표를 target month로 이동한 implied future score입니다. "
                    f"실제 관측 구간은 {_format_date(actual_cutoff)}까지입니다."
                )
        else:
            st.info("승인된 선행 pair가 3개 미만이라 composite score를 표시하지 않습니다.")
        st.caption(f"Approved leading pairs: {len(approved)} / {len(reports)}")
    with summary_columns[1]:
        st.markdown('<span class="section-chip">Current Nowcast Snapshot</span>', unsafe_allow_html=True)
        if month_nowcast.ok and not nowcast_snapshot.empty:
            snapshot = nowcast_snapshot.copy()
            snapshot["value"] = snapshot["value"].map(lambda value: _format_value(value))
            st.dataframe(snapshot, width="stretch", hide_index=True)
        else:
            st.warning("Cleveland monthly nowcast가 비어 있습니다.")


def render_leading_signals(reports: list[ValidationReport]) -> None:
    st.header("Leading Signals")
    approved_reports = [report for report in reports if report.approved]
    watchlist_reports = _watchlist_reports(reports)
    st.caption(f"승인 {len(approved_reports)}개 / 전체 {len(reports)}개 pair")
    if not approved_reports:
        st.warning("검증 기준을 통과한 선행지표가 아직 없습니다. 하단 Data QA에서 거절 사유를 확인해 주세요.")

    for report in approved_reports:
        title = (
            f"{report.title} | lag {report.lag_months}m | corr {report.full_corr:.2f}"
            if report.full_corr is not None
            else report.title
        )
        with st.expander(title, expanded=False):
            stat_columns = st.columns(4)
            stat_columns[0].metric("Lag", f"{report.lag_months}m")
            stat_columns[1].metric("Aligned Corr", f"{report.full_corr:.2f}" if report.full_corr is not None else "N/A")
            stat_columns[2].metric("Rolling 24M", f"{report.rolling_corr_24:.2f}" if report.rolling_corr_24 is not None else "N/A")
            horizon = _format_date(pd.to_datetime(report.projection_frame["date"]).max()) if not report.projection_frame.empty else "N/A"
            stat_columns[3].metric("Projection To", horizon)

            aligned_view = st.toggle("Aligned View", value=True, key=f"aligned_{report.key}")
            st.plotly_chart(
                build_leading_figure(report, aligned_view=aligned_view),
                width="stretch",
                config=PLOTLY_CONFIG,
            )
            if aligned_view and not report.projection_frame.empty:
                st.caption(
                    f"점선은 {_format_date(report.latest_target_date)} 이후의 implied future path입니다. "
                    f"{report.indicator_label}를 {report.lag_months}개월 앞으로 이동한 값입니다."
                )
                projection_preview = _projection_preview_frame(report, months=6)
                if not projection_preview.empty:
                    st.dataframe(projection_preview, width="stretch", hide_index=True)
            if report.freshness_note:
                st.caption(f"Freshness note: {report.freshness_note}. stale source도 경고와 함께 계속 표시합니다.")
            st.caption(report.note)
            st.caption(
                f"{report.indicator_label}: {report.source_urls.get(report.indicator_key, '')} | "
                f"{report.target_label}: {report.source_urls.get(report.target_key, '')}"
            )

    if watchlist_reports:
        st.subheader("PPI Watchlist")
        st.caption(
            "아래 PPI pair는 데이터가 없어서 빠진 것이 아니라, 최근 24개월 상관 약화나 lag 안정성 문제로 승인 기준에서만 탈락한 상태입니다. "
            "해석 가치는 남아 있으므로 경고와 함께 계속 표시합니다."
        )
        for report in watchlist_reports:
            title = (
                f"{report.title} | watchlist | lag {report.lag_months}m | hist corr {report.full_corr:.2f}"
                if report.full_corr is not None
                else f"{report.title} | watchlist"
            )
            with st.expander(title, expanded=False):
                stat_columns = st.columns(4)
                stat_columns[0].metric("Lag", f"{report.lag_months}m" if report.lag_months is not None else "N/A")
                stat_columns[1].metric("Hist Corr", f"{report.full_corr:.2f}" if report.full_corr is not None else "N/A")
                stat_columns[2].metric("Rolling 24M", f"{report.rolling_corr_24:.2f}" if report.rolling_corr_24 is not None else "N/A")
                horizon = _format_date(pd.to_datetime(report.projection_frame["date"]).max()) if not report.projection_frame.empty else "N/A"
                stat_columns[3].metric("Projection To", horizon)

                aligned_view = st.toggle("Aligned View", value=True, key=f"watchlist_aligned_{report.key}")
                st.plotly_chart(
                    build_leading_figure(report, aligned_view=aligned_view),
                    width="stretch",
                    config=PLOTLY_CONFIG,
                )
                if report.reasons:
                    st.warning(f"Watchlist reason: {'; '.join(report.reasons)}")
                if aligned_view and not report.projection_frame.empty:
                    st.caption(
                        f"점선은 {_format_date(report.latest_target_date)} 이후의 implied future path입니다. "
                        f"{report.indicator_label}를 {report.lag_months}개월 앞으로 이동한 값입니다."
                    )
                    projection_preview = _projection_preview_frame(report, months=6)
                    if not projection_preview.empty:
                        st.dataframe(projection_preview, width="stretch", hide_index=True)
                if report.freshness_note:
                    st.caption(f"Freshness note: {report.freshness_note}. stale source도 경고와 함께 계속 표시합니다.")
                st.caption("Historical relationship is still strong, but the recent regime check failed.")
                st.caption(
                    f"{report.indicator_label}: {report.source_urls.get(report.indicator_key, '')} | "
                    f"{report.target_label}: {report.source_urls.get(report.target_key, '')}"
                )


def render_nowcast_persistence(sources: dict[str, SourceResult]) -> None:
    st.header("Nowcast & Persistence")
    st.caption("이 섹션의 지표는 현재 추정/점착성 확인용이며, composite leading score에는 포함하지 않습니다.")

    monthly_nowcast = sources["cleveland_nowcast_month"]
    quarter_nowcast = sources["cleveland_nowcast_quarter"]
    sticky = sources["atlanta_sticky_cpi"]
    wage = sources["atlanta_wage_growth"]

    top_columns = st.columns([1.5, 1.0])
    with top_columns[0]:
        if monthly_nowcast.ok:
            st.plotly_chart(build_nowcast_figure(monthly_nowcast), width="stretch", config=PLOTLY_CONFIG)
        else:
            st.warning(f"Cleveland monthly nowcast unavailable: {monthly_nowcast.detail}")
    with top_columns[1]:
        if monthly_nowcast.ok:
            snapshot = _current_nowcast_snapshot(monthly_nowcast.frame)
            snapshot["value"] = snapshot["value"].map(lambda value: _format_value(value))
            st.dataframe(snapshot, width="stretch", hide_index=True)
            st.caption(f"Monthly URL: {monthly_nowcast.source_url}")
        else:
            st.info("Monthly nowcast snapshot unavailable.")

        if quarter_nowcast.ok and not quarter_nowcast.frame.empty:
            quarter_display = quarter_nowcast.frame.copy()
            for column in ("cpi", "core_cpi", "pce", "core_pce"):
                if column in quarter_display.columns:
                    quarter_display[column] = quarter_display[column].map(lambda value: _format_value(value))
            st.dataframe(quarter_display, width="stretch", hide_index=True)

    if sticky.ok or wage.ok:
        st.plotly_chart(build_persistence_figure(sticky, wage), width="stretch", config=PLOTLY_CONFIG)
        st.caption(f"Sticky CPI URL: {sticky.source_url} | Wage Growth URL: {wage.source_url}")
    else:
        st.warning("Atlanta persistence series are unavailable.")


def render_data_qa(sources: dict[str, SourceResult], reports: list[ValidationReport]) -> None:
    st.header("Data QA")
    source_status = build_source_status_frame(sources)
    rejected = build_rejected_frame(reports)

    st.subheader("Source Status")
    st.dataframe(source_status, width="stretch", hide_index=True)

    st.subheader("Rejected / QA")
    if rejected.empty:
        st.success("현재 모든 leading pair가 승인되었습니다.")
    else:
        st.dataframe(rejected, width="stretch", hide_index=True)

    stale_sources = source_status.loc[source_status["status"] == "stale", "source"].tolist()
    failed_sources = source_status.loc[source_status["status"] == "failed", "source"].tolist()
    if stale_sources:
        st.warning(f"Stale sources: {', '.join(stale_sources)}")
    if failed_sources:
        st.error(f"Failed sources: {', '.join(failed_sources)}")


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=":bar_chart:",
        layout="wide",
    )
    _inject_styles()

    st.markdown(
        f"""
        <div class="hero">
            <div class="section-chip">US CPI Focus</div>
            <h1 style="margin:0.45rem 0 0.25rem 0;">{APP_TITLE}</h1>
            <p style="margin:0;color:#4b5b6a;">{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "refresh_nonce" not in st.session_state:
        st.session_state["refresh_nonce"] = 0

    control_columns = st.columns([1, 3])
    with control_columns[0]:
        if st.button("Refresh Data", width="stretch"):
            load_monthly_sources.clear()
            load_cleveland_sources.clear()
            _delete_snapshot(MONTHLY_SNAPSHOT_PATH)
            _delete_snapshot(CLEVELAND_SNAPSHOT_PATH)
            st.session_state["refresh_nonce"] += 1
            st.rerun()
    with control_columns[1]:
        if _fixture_mode():
            st.caption("Fixture mode is enabled for deterministic UI/test rendering.")
        else:
            st.caption("Warm rerun uses Streamlit cache. Cold start falls back to recent local snapshot files when available.")

    sources = load_all_sources(refresh_token=st.session_state["refresh_nonce"])
    reports = build_validation_reports(sources)
    forward_score, forward_frame, forward_note = compute_forward_pressure(reports)

    render_overview(sources, reports, forward_score, forward_frame, forward_note)
    render_leading_signals(reports)
    render_nowcast_persistence(sources)
    render_data_qa(sources, reports)


if __name__ == "__main__":
    main()
