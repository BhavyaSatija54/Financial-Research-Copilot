"""
FRED (Federal Reserve Economic Data) fetcher.

Fetches macroeconomic series from the St. Louis Fed API and converts
them to text documents suitable for RAG ingestion.

Requires: FRED_API_KEY in .env
Fallback: If no API key, uses public FRED CSV endpoint (limited series).

Key series covered:
  GDP, CPI, Core PCE, Federal Funds Rate, Unemployment Rate,
  10Y Treasury Yield, 2Y Treasury Yield, Industrial Production,
  Retail Sales, Consumer Confidence, Housing Starts, M2 Money Supply
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

# Series to fetch: (series_id, human_name, category)
MACRO_SERIES: list[tuple[str, str, str]] = [
    ("GDP",       "US Gross Domestic Product",                    "growth"),
    ("CPIAUCSL",  "Consumer Price Index (All Urban Consumers)",   "inflation"),
    ("PCEPILFE",  "Core PCE Price Index",                         "inflation"),
    ("FEDFUNDS",  "Federal Funds Effective Rate",                 "monetary_policy"),
    ("UNRATE",    "Civilian Unemployment Rate",                   "labour_market"),
    ("GS10",      "10-Year Treasury Constant Maturity Rate",      "rates"),
    ("GS2",       "2-Year Treasury Constant Maturity Rate",       "rates"),
    ("T10Y2Y",    "10-Year minus 2-Year Treasury Spread",         "rates"),
    ("INDPRO",    "Industrial Production Index",                  "activity"),
    ("RSAFS",     "Advance Retail Sales",                         "activity"),
    ("UMCSENT",   "University of Michigan Consumer Sentiment",    "sentiment"),
    ("HOUST",     "Housing Starts",                               "housing"),
    ("M2SL",      "M2 Money Stock",                               "monetary_policy"),
    ("DEXUSEU",   "USD/EUR Exchange Rate",                        "fx"),
    ("DEXJPUS",   "JPY/USD Exchange Rate",                        "fx"),
]

FRED_BASE = "https://api.stlouisfed.org/fred"
FRED_CSV_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


@dataclass
class MacroDocument:
    """A macro series formatted as a RAG-ingestible text document."""

    doc_id: str
    title: str
    series_id: str
    category: str
    text: str
    period_start: str
    period_end: str
    metadata: dict[str, Any]


class FREDFetcher:
    """
    Async FRED data fetcher.

    With API key: uses official FRED JSON API (full history, all series).
    Without API key: falls back to public CSV endpoint (last 10 years).
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "FREDFetcher":
        self._session = aiohttp.ClientSession(
            headers={"Accept-Encoding": "gzip"},
            timeout=aiohttp.ClientTimeout(total=30),
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session:
            await self._session.close()

    async def fetch_all(
        self,
        series: list[tuple[str, str, str]] | None = None,
        observation_start: str = "2018-01-01",
    ) -> list[MacroDocument]:
        """
        Fetch all configured macro series and return as documents.

        Parameters
        ----------
        series : list of (series_id, name, category) or None for defaults
        observation_start : ISO date string

        Returns
        -------
        list[MacroDocument]
        """
        targets = series or MACRO_SERIES
        log.info("fred_fetch_start", n_series=len(targets), api_key=bool(self._api_key))

        tasks = [
            self._fetch_series(sid, name, cat, observation_start)
            for sid, name, cat in targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: list[MacroDocument] = []
        for (sid, name, _), result in zip(targets, results):
            if isinstance(result, Exception):
                log.warning("fred_series_error", series=sid, error=str(result))
            elif result is not None:
                docs.append(result)

        log.info("fred_fetch_done", fetched=len(docs), failed=len(targets) - len(docs))
        return docs

    async def _fetch_series(
        self,
        series_id: str,
        name: str,
        category: str,
        observation_start: str,
    ) -> MacroDocument | None:
        """Fetch a single series and format it as a document."""
        try:
            if self._api_key:
                df = await self._fetch_via_api(series_id, observation_start)
            else:
                df = await self._fetch_via_csv(series_id, observation_start)

            if df is None or df.empty:
                log.warning("fred_empty_series", series=series_id)
                return None

            text = self._format_series_as_text(df, series_id, name, category)
            period_start = str(df.index[0].date())
            period_end = str(df.index[-1].date())
            today = datetime.now().strftime("%Y-%m-%d")

            return MacroDocument(
                doc_id=f"FRED_{series_id}_{today}",
                title=f"FRED Series: {name} ({series_id})",
                series_id=series_id,
                category=category,
                text=text,
                period_start=period_start,
                period_end=period_end,
                metadata={
                    "ticker": None,
                    "company": "Federal Reserve Economic Data",
                    "filing_type": "MACRO",
                    "doc_type": "macro",
                    "series_id": series_id,
                    "category": category,
                    "period": f"{period_start} to {period_end}",
                    "filed_date": today,
                    "source": "FRED",
                    "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
                },
            )
        except Exception as exc:
            log.error("fred_fetch_error", series=series_id, error=str(exc))
            return None

    async def _fetch_via_api(self, series_id: str, observation_start: str) -> pd.DataFrame | None:
        assert self._session and self._api_key
        url = f"{FRED_BASE}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": observation_start,
            "sort_order": "asc",
        }
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()

        observations = data.get("observations", [])
        if not observations:
            return None

        records = []
        for obs in observations:
            try:
                val = float(obs["value"])
                records.append({"date": obs["date"], "value": val})
            except (ValueError, KeyError):
                continue  # skip missing values (".")

        if not records:
            return None

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        return df

    async def _fetch_via_csv(self, series_id: str, observation_start: str) -> pd.DataFrame | None:
        """Public CSV fallback — no API key needed."""
        assert self._session
        url = f"{FRED_CSV_BASE}?id={series_id}"
        async with self._session.get(url) as resp:
            if resp.status != 200:
                return None
            text = await resp.text()

        from io import StringIO
        try:
            df = pd.read_csv(StringIO(text), parse_dates=["DATE"], index_col="DATE")
            df.columns = ["value"]
            df = df[df["value"] != "."]
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna()
            cutoff = pd.Timestamp(observation_start)
            df = df[df.index >= cutoff]
            return df if not df.empty else None
        except Exception:
            return None

    def _format_series_as_text(
        self,
        df: pd.DataFrame,
        series_id: str,
        name: str,
        category: str,
    ) -> str:
        """Convert a DataFrame of observations to a rich text document."""
        latest = df.iloc[-1]["value"]
        prev_year = df.last("365D").iloc[0]["value"] if len(df) > 1 else latest
        yoy_change = ((latest - prev_year) / abs(prev_year) * 100) if prev_year != 0 else 0.0

        # Compute trailing stats
        recent_12m = df.last("365D")["value"]
        mean_12m = recent_12m.mean()
        min_12m = recent_12m.min()
        max_12m = recent_12m.max()

        # Recent observations table
        last_obs = df.tail(24)
        obs_lines = "\n".join(
            f"  {date.strftime('%Y-%m-%d')}: {val:.4f}"
            for date, val in zip(last_obs.index, last_obs["value"])
        )

        return f"""FRED MACROECONOMIC DATA REPORT
Series: {series_id}
Name: {name}
Category: {category.replace('_', ' ').title()}
Source: Federal Reserve Bank of St. Louis (FRED)
Data Range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}

LATEST READING
  Value: {latest:.4f}
  Date: {df.index[-1].strftime('%Y-%m-%d')}
  Year-over-Year Change: {yoy_change:+.2f}%

TRAILING 12-MONTH STATISTICS
  Mean: {mean_12m:.4f}
  Minimum: {min_12m:.4f}
  Maximum: {max_12m:.4f}
  Range: {max_12m - min_12m:.4f}

RECENT OBSERVATIONS (Last 24 data points)
{obs_lines}

INTERPRETATION NOTES
{self._interpretation_notes(series_id, latest, yoy_change)}

For full historical data and chart, visit: https://fred.stlouisfed.org/series/{series_id}
"""

    @staticmethod
    def _interpretation_notes(series_id: str, latest: float, yoy_change: float) -> str:
        notes = {
            "GDP": (
                f"US real GDP growth of {yoy_change:+.2f}% year-over-year. "
                "Readings above 2% are generally considered healthy expansion. "
                "Negative readings for two consecutive quarters signal a technical recession."
            ),
            "CPIAUCSL": (
                f"Headline CPI inflation at {latest:.2f}, up {yoy_change:+.2f}% year-over-year. "
                "The Fed targets 2% PCE inflation. Elevated CPI may prompt rate hikes."
            ),
            "PCEPILFE": (
                f"Core PCE (ex food & energy) at {latest:.2f}, the Fed's preferred inflation gauge. "
                f"Year-over-year change: {yoy_change:+.2f}%. Target is 2.0%."
            ),
            "FEDFUNDS": (
                f"Federal funds effective rate at {latest:.2f}%. "
                "This is the benchmark overnight lending rate set by FOMC. "
                "Higher rates tighten financial conditions and increase borrowing costs."
            ),
            "UNRATE": (
                f"Unemployment rate at {latest:.1f}%. "
                "Full employment is generally considered 4.0-4.5%. "
                "Below 4% may signal labour market tightness and wage pressure."
            ),
            "GS10": (
                f"10-year Treasury yield at {latest:.2f}%. "
                "This is the benchmark risk-free rate used to discount equities and price mortgages."
            ),
            "T10Y2Y": (
                f"Yield curve spread (10Y-2Y) at {latest:.2f}%. "
                "Negative readings (inversion) historically precede recessions by 12-18 months."
            ),
        }
        return notes.get(
            series_id,
            f"Current reading: {latest:.4f}. Year-over-year change: {yoy_change:+.2f}%.",
        )
