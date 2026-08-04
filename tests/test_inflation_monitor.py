"""Regression tests for the pure data-processing logic in inflation_monitor.

These tests run fully offline (no network access) so they can catch
regressions in parsing/analytics code independent of whether the scraped
websites are reachable or have changed their markup.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inflation_monitor as im


# ---------------------------------------------------------------------------
# Monthly frame normalization
# ---------------------------------------------------------------------------


def test_normalize_monthly_frame_dedupes_by_latest_release():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-15", "2024-01-20", "2024-02-01"],
            "value": [100.0, 101.0, 102.0],
            "release_date": ["2024-02-01", "2024-02-05", "2024-03-01"],
        }
    )
    frame = im._normalize_monthly_frame(raw)
    assert list(frame["date"].dt.strftime("%Y-%m")) == ["2024-01", "2024-02"]
    # Keeps the row with the most recent release_date for a duplicated month.
    assert frame.loc[frame["date"] == pd.Timestamp("2024-01-01"), "value"].iloc[0] == 101.0


def test_normalize_monthly_frame_drops_missing_markers():
    raw = pd.DataFrame({"date": ["2024-01-01", "2024-02-01"], "value": [".", "5.0"]})
    frame = im._normalize_monthly_frame(raw)
    assert len(frame) == 1
    assert frame["value"].iloc[0] == 5.0


def test_normalize_monthly_frame_empty_input():
    frame = im._normalize_monthly_frame(pd.DataFrame())
    assert list(frame.columns) == ["date", "value", "release_date"]
    assert frame.empty


# ---------------------------------------------------------------------------
# Transforms / lag alignment
# ---------------------------------------------------------------------------


def _monthly_series(start: str, values: list[float]) -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq="MS")
    return pd.Series(values, index=index)


def test_transform_series_yoy():
    values = [100.0 * (1.01**i) for i in range(24)]
    series = _monthly_series("2022-01-01", values)
    yoy = im._transform_series(series, "yoy")
    # 12 months later, growth compounds to ~1.01**12 - 1 => ~12.68%
    assert yoy.dropna().iloc[0] == pytest.approx((1.01**12 - 1.0) * 100.0, rel=1e-6)


def test_shift_forward_moves_index():
    series = _monthly_series("2024-01-01", [1.0, 2.0, 3.0])
    shifted = im._shift_forward(series, 2)
    assert shifted.index[0] == pd.Timestamp("2024-03-01")


def test_choose_best_lag_recovers_known_lag():
    rng = np.random.default_rng(0)
    base = pd.Series(np.sin(np.linspace(0, 20, 96)), index=pd.date_range("2016-01-01", periods=96, freq="MS"))
    indicator = base + rng.normal(0, 0.01, size=96)
    target = base.shift(3).bfill() + rng.normal(0, 0.01, size=96)

    result = im._choose_best_lag(indicator, target, lag_min=0, lag_max=6)
    assert result["best_train"] is not None
    assert result["best_train"]["lag"] == 3


# ---------------------------------------------------------------------------
# validate_indicator / build_validation_reports using synthetic sources
# ---------------------------------------------------------------------------


def _synthetic_source(key: str, label: str, *, lag: int = 0, n: int = 96) -> im.SourceResult:
    end = im._utc_now().to_period("M").to_timestamp()
    dates = pd.date_range(end=end, periods=n, freq="MS")
    signal = np.sin(np.linspace(0, 24, n))
    level = 100.0 * np.exp(np.cumsum(0.002 + 0.001 * np.roll(signal, lag)))
    frame = im._normalize_monthly_frame(pd.DataFrame({"date": dates, "value": level}))
    return im._finalize_source_result(key, label, frame, f"https://example.test/{key}")


def test_validate_indicator_approves_strong_relationship():
    spec = im.IndicatorSpec(
        key="a_to_b",
        title="A -> B",
        indicator_key="indicator_a",
        target_key="target_b",
        indicator_transform="yoy",
        target_transform="yoy",
        lag_min=0,
        lag_max=6,
    )
    sources = {
        "indicator_a": _synthetic_source("indicator_a", "Indicator A", lag=0),
        "target_b": _synthetic_source("target_b", "Target B", lag=3),
    }
    report = im.validate_indicator(spec, sources)
    assert report.approved, report.reasons


def test_validate_indicator_rejects_missing_source():
    spec = im.IndicatorSpec(
        key="a_to_missing",
        title="A -> Missing",
        indicator_key="indicator_a",
        target_key="does_not_exist",
    )
    sources = {"indicator_a": _synthetic_source("indicator_a", "Indicator A")}
    report = im.validate_indicator(spec, sources)
    assert not report.approved
    assert "missing source definition" in report.reasons


def test_validate_indicator_rejects_failed_source():
    spec = im.IndicatorSpec(
        key="a_to_b",
        title="A -> B",
        indicator_key="indicator_a",
        target_key="target_b",
    )
    sources = {
        "indicator_a": _synthetic_source("indicator_a", "Indicator A"),
        "target_b": im._failed_source_result("target_b", "Target B", "https://example.test", "boom"),
    }
    report = im.validate_indicator(spec, sources)
    assert not report.approved
    assert any("target unavailable" in reason for reason in report.reasons)


def test_compute_forward_pressure_withheld_below_threshold():
    report = im.ValidationReport(
        key="only_one",
        title="Only One",
        indicator_key="a",
        target_key="b",
        indicator_label="A",
        target_label="B",
        approved=True,
        score_weight=0.5,
        shifted_indicator_frame=pd.DataFrame({"date": pd.date_range("2020-01-01", periods=30, freq="MS"), "indicator": range(30)}),
    )
    score, frame, note = im.compute_forward_pressure([report])
    assert score is None
    assert frame.empty
    assert "withheld" in note


# ---------------------------------------------------------------------------
# NBS (China PPI) text parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "mode", "expected"),
    [
        (
            "The producer price index for industrial products (PPI) increased by 2.3 percent year-on-year.",
            "yoy",
            2.3,
        ),
        (
            "The producer price index for industrial products (PPI) decreased by 1.8 percent year-on-year.",
            "yoy",
            -1.8,
        ),
        (
            "It remained unchanged month-on-month.",
            "mom",
            0.0,
        ),
        (
            "The producer price index for industrial products (PPI) increased by 0.3 percent month-on-month.",
            "mom",
            0.3,
        ),
    ],
)
def test_parse_nbs_paragraph_metric(text, mode, expected):
    assert im._try_parse_nbs_paragraph_metric(text, mode) == pytest.approx(expected)


def test_is_nbs_ppi_title_matches_known_variants():
    assert im._is_nbs_ppi_title("Industrial Producer Price Indexes in December 2023")
    assert im._is_nbs_ppi_title("3. Producer Prices in the Industrial Sector for January 2024")
    assert not im._is_nbs_ppi_title("Consumer Price Index for December 2023")


def test_parse_observation_month():
    assert im._parse_observation_month("Industrial Producer Price Indexes in March 2024") == pd.Timestamp(
        "2024-03-01"
    )


# ---------------------------------------------------------------------------
# ISM prices-paid text parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The Prices Index registered 65.4 percent in March, an increase from the prior month.",
        "The Prices Index was 65.4 percent, up from February's reading.",
        "Manufacturers reported prices paid 65.4 percent in the survey.",
    ],
)
def test_parse_ism_prices_from_text(text):
    assert im._parse_ism_prices_from_text(text) == pytest.approx(65.4)


def test_parse_ism_prices_from_text_returns_none_when_absent():
    assert im._parse_ism_prices_from_text("No relevant figures here.") is None


# ---------------------------------------------------------------------------
# fetch_fred_series honors the shared session and reports true TLS fallback
# ---------------------------------------------------------------------------


def test_fetch_fred_series_uses_shared_session_and_reports_real_tls_state():
    csv_text = "DATE,CPIAUCSL\n2024-01-01,300.1\n2024-02-01,301.2\n"

    with patch.object(im, "_request_text", return_value=(csv_text, False)) as mocked:
        result = im.fetch_fred_series(
            im._new_session(), key="headline_cpi", series_id="CPIAUCSL", label="US CPI"
        )

    mocked.assert_called_once()
    assert result.insecure_tls is False
    assert result.ok
    assert result.row_count == 2


def test_fetch_fred_series_propagates_true_tls_fallback():
    csv_text = "DATE,CPIAUCSL\n2024-01-01,300.1\n"
    with patch.object(im, "_request_text", return_value=(csv_text, True)):
        result = im.fetch_fred_series(
            im._new_session(), key="headline_cpi", series_id="CPIAUCSL", label="US CPI"
        )
    assert result.insecure_tls is True


# ---------------------------------------------------------------------------
# NBS archive-page fetch degrades gracefully instead of crashing everything
# ---------------------------------------------------------------------------


def test_fetch_nbs_archive_page_swallows_request_errors():
    with patch.object(im, "_request", side_effect=im.requests.exceptions.ConnectionError("boom")):
        page_number, links, insecure = im._fetch_nbs_archive_page(3)
    assert page_number == 3
    assert links == []
    assert insecure is False


def test_fetch_china_ppi_yoy_survives_broken_search_endpoint():
    with patch.object(im, "_iter_nbs_search_results", side_effect=ValueError("bootstrap not found")), \
         patch.object(im, "_iter_nbs_archive_links", return_value=([], False)):
        result = im.fetch_china_ppi_yoy(im._new_session())
    # Should not raise, and should degrade to an empty-but-well-formed result.
    assert isinstance(result, im.SourceResult)
    assert result.key == "china_ppi_yoy"


# ---------------------------------------------------------------------------
# ISM Prices Paid is wired into the app's monthly source set and leading specs
# ---------------------------------------------------------------------------


def test_ism_prices_paid_is_registered_as_a_monthly_source():
    assert "ism_prices_paid" in im.MONTHLY_SOURCE_KEYS
    assert "ism_prices_paid" in im.DISPLAY_LABELS


def test_ism_prices_paid_has_a_leading_indicator_spec():
    keys = {spec.indicator_key for spec in im.LEADING_SPECS}
    assert "ism_prices_paid" in keys


def test_fixture_sources_include_ism_prices_paid():
    fixtures = im._build_fixture_sources()
    assert "ism_prices_paid" in fixtures
    assert fixtures["ism_prices_paid"].ok


# ---------------------------------------------------------------------------
# ISM direct-page fetch must not mislabel prior-year months with this year
# ---------------------------------------------------------------------------


def test_fetch_ism_direct_pages_uses_requested_period_not_response_url():
    """A response URL like '.../pmi/september/' carries no year. Prior-year
    months walked in a months_back window must not be stamped with the
    current year just because the URL lacks one."""

    class _FakeResponse:
        def __init__(self, url: str, text: str) -> None:
            self.url = url
            self.text = text

    def fake_request(session, url, **kwargs):
        # No year segment in the URL, mirroring the real ISM site's direct pages.
        return _FakeResponse(url, "The Prices Index registered 58.0 percent."), False

    with patch.object(im, "_request", side_effect=fake_request):
        frame, insecure, failures, blocked = im._fetch_ism_direct_pages(im._new_session(), months_back=12)

    assert not blocked
    assert failures == 0
    assert not frame.empty
    current_year = im._utc_now().year
    # A 12-month lookback from "now" must span two different years
    # (the whole point of this regression test).
    assert frame["date"].dt.year.nunique() == 2
    assert set(frame["date"].dt.year) == {current_year - 1, current_year}


# ---------------------------------------------------------------------------
# ISM roundup (sitemap) failures must not discard already-fetched direct rows
# ---------------------------------------------------------------------------


def test_request_does_not_retry_plain_timeout():
    """A plain read timeout is not a TLS problem, so retrying with
    verify=False would just wait out a second identical timeout for
    nothing. It should propagate immediately instead of doubling the wait."""
    session = im._new_session()
    with patch.object(session, "request", side_effect=im.requests.exceptions.ReadTimeout("boom")) as mocked:
        with pytest.raises(im.requests.exceptions.ReadTimeout):
            im._request(session, "https://example.test")
    assert mocked.call_count == 1


def test_request_retries_once_on_ssl_error():
    session = im._new_session()
    ok_response = MagicMock()
    ok_response.raise_for_status.return_value = None
    verify_flags: list[bool] = []

    def fake_request(method, url, **kwargs):
        verify_flags.append(kwargs.get("verify"))
        if kwargs.get("verify"):
            raise im.requests.exceptions.SSLError("cert issue")
        return ok_response

    with patch.object(session, "request", side_effect=fake_request):
        response, insecure = im._request(session, "https://example.test")

    assert insecure is True
    assert verify_flags == [True, False]


# ---------------------------------------------------------------------------
# FRED series are fetched concurrently, not one-by-one
# ---------------------------------------------------------------------------


def test_load_monthly_sources_fetches_all_fred_series(monkeypatch, tmp_path):
    monkeypatch.setattr(im, "MONTHLY_SNAPSHOT_PATH", tmp_path / "monthly.pkl")
    monkeypatch.delenv("INFLATION_MONITOR_FIXTURE_MODE", raising=False)

    def fake_fetch_fred_series(session, *, key, series_id, label):
        frame = im._normalize_monthly_frame(
            pd.DataFrame({"date": pd.date_range("2024-01-01", periods=2, freq="MS"), "value": [1.0, 2.0]})
        )
        return im._finalize_source_result(key, label, frame, "https://example.test")

    def fake_other_loader(key, label):
        return lambda session: im._failed_source_result(key, label, "https://example.test", "skipped in test")

    with patch.object(im, "fetch_fred_series", side_effect=fake_fetch_fred_series), \
         patch.object(im, "fetch_fao_food_price_index", side_effect=fake_other_loader("fao_food_price_index", "FAO")), \
         patch.object(im, "fetch_atlanta_wage_growth", side_effect=fake_other_loader("atlanta_wage_growth", "AWG")), \
         patch.object(im, "fetch_atlanta_sticky_cpi", side_effect=fake_other_loader("atlanta_sticky_cpi", "ASC")), \
         patch.object(im, "fetch_china_ppi_yoy", side_effect=fake_other_loader("china_ppi_yoy", "CPPI")), \
         patch.object(im, "fetch_ism_prices_paid", side_effect=fake_other_loader("ism_prices_paid", "ISM")):
        im.load_monthly_sources.clear()
        payloads = im.load_monthly_sources(force_live=True)

    assert set(im.FRED_SERIES.keys()).issubset(payloads.keys())
    for key in im.FRED_SERIES:
        assert payloads[key]["ok"], payloads[key]["detail"]


def test_fetch_ism_prices_paid_keeps_direct_rows_when_roundup_fails():
    direct_frame = im._normalize_monthly_frame(
        pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="MS"),
                "value": [58.0, 59.0, 60.0],
            }
        )
    )
    with patch.object(im, "_fetch_ism_direct_pages", return_value=(direct_frame, False, 0, False)), \
         patch.object(im, "_fetch_ism_roundup_pages", side_effect=im.requests.exceptions.ConnectionError("sitemap down")):
        result = im.fetch_ism_prices_paid(im._new_session())

    assert result.ok
    assert result.row_count == 3
