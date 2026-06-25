"""
SEC EDGAR document fetcher.

Supports:
  - Fetching 10-K, 10-Q, 8-K filings by ticker / CIK
  - Full-text extraction from HTML/XBRL filings
  - Rate-limited requests respecting EDGAR's fair-use policy
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import aiohttp
from bs4 import BeautifulSoup

from src.utils.config import get_settings
from src.utils.logger import get_logger

log = get_logger(__name__)

FilingType = Literal["10-K", "10-Q", "8-K", "DEF 14A", "S-1"]

EDGAR_BASE = "https://data.sec.gov"
FULL_TEXT_SEARCH = "https://efts.sec.gov/LATEST/search-index"
COMPANY_FACTS_URL = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{{cik:010d}}.json"
SUBMISSIONS_URL = f"{EDGAR_BASE}/submissions/CIK{{cik:010d}}.json"


@dataclass
class SECFiling:
    """Parsed SEC filing document."""

    ticker: str
    cik: str
    filing_type: str
    accession_number: str
    filed_date: str
    period_of_report: str
    company_name: str
    text: str
    source_url: str
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return f"{self.ticker}_{self.filing_type}_{self.period_of_report}".replace("/", "-")


class SECFetcher:
    """Async SEC EDGAR client."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._session: aiohttp.ClientSession | None = None
        self._last_request: float = 0.0

    async def __aenter__(self) -> "SECFetcher":
        headers = {"User-Agent": self.settings.sec_user_agent}
        self._session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._session:
            await self._session.close()

    async def _get(self, url: str) -> dict | str:
        """Rate-limited GET request."""
        assert self._session, "Use as async context manager"
        elapsed = time.monotonic() - self._last_request
        delay = self.settings.sec_request_delay_seconds
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)

        async with self._session.get(url) as resp:
            self._last_request = time.monotonic()
            resp.raise_for_status()
            ct = resp.content_type
            if "json" in ct:
                return await resp.json()
            return await resp.text()

    async def get_cik(self, ticker: str) -> str:
        """Resolve ticker symbol to CIK."""
        url = f"{EDGAR_BASE}/cgi-bin/browse-edgar?action=getcompany&company={ticker}&CIK=&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany"
        # Use the company search JSON endpoint
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
        data = await self._get(
            f"https://www.sec.gov/cgi-bin/browse-edgar?company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=5&search_text=&action=getcompany&output=atom"
        )
        # Extract CIK from EDGAR search response
        if isinstance(data, str):
            match = re.search(r"CIK=(\d+)", data)
            if match:
                return match.group(1).zfill(10)
        raise ValueError(f"Could not resolve CIK for ticker: {ticker}")

    async def get_filings(
        self,
        ticker: str,
        filing_type: FilingType = "10-K",
        limit: int = 5,
    ) -> list[SECFiling]:
        """Fetch and parse SEC filings for a given ticker."""
        log.info("sec_fetch_start", ticker=ticker, filing_type=filing_type, limit=limit)

        try:
            cik = await self.get_cik(ticker)
        except Exception as exc:
            log.error("cik_resolution_failed", ticker=ticker, error=str(exc))
            raise

        submissions_url = SUBMISSIONS_URL.format(cik=int(cik))
        submissions = await self._get(submissions_url)
        assert isinstance(submissions, dict)

        company_name = submissions.get("name", ticker)
        recent = submissions.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates = recent.get("filingDate", [])
        periods = recent.get("reportDate", [])

        filings: list[SECFiling] = []
        count = 0

        for form, accession, date, period in zip(forms, accessions, dates, periods):
            if count >= limit:
                break
            if form != filing_type:
                continue

            try:
                text = await self._fetch_filing_text(cik, accession)
                filing = SECFiling(
                    ticker=ticker,
                    cik=cik,
                    filing_type=form,
                    accession_number=accession,
                    filed_date=date,
                    period_of_report=period,
                    company_name=company_name,
                    text=text,
                    source_url=self._accession_url(cik, accession),
                    metadata={
                        "ticker": ticker,
                        "company": company_name,
                        "filing_type": form,
                        "filed_date": date,
                        "period": period,
                        "source": "SEC_EDGAR",
                    },
                )
                filings.append(filing)
                count += 1
                log.info(
                    "filing_fetched",
                    ticker=ticker,
                    filing_type=form,
                    period=period,
                    text_len=len(text),
                )
            except Exception as exc:
                log.warning(
                    "filing_fetch_error",
                    ticker=ticker,
                    accession=accession,
                    error=str(exc),
                )

        return filings

    async def _fetch_filing_text(self, cik: str, accession: str) -> str:
        """Download and extract plain text from a filing."""
        acc_clean = accession.replace("-", "")
        index_url = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"

        try:
            index_html = await self._get(index_url)
            assert isinstance(index_html, str)
            doc_url = self._extract_primary_doc_url(cik, acc_clean, index_html)
            doc_html = await self._get(doc_url)
            assert isinstance(doc_html, str)
            return self._html_to_text(doc_html)
        except Exception:
            # Fallback: try the .txt full submission file
            txt_url = (
                f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}.txt"
            )
            raw = await self._get(txt_url)
            assert isinstance(raw, str)
            return self._clean_text(raw)

    def _extract_primary_doc_url(self, cik: str, acc_clean: str, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        base = f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/"
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) >= 3:
                doc_type = cells[3].get_text(strip=True) if len(cells) > 3 else ""
                href = cells[2].find("a")
                if href and doc_type in ("10-K", "10-Q", "8-K", ""):
                    return base + href["href"]
        raise ValueError("Primary document not found in filing index")

    def _accession_url(self, cik: str, accession: str) -> str:
        acc_clean = accession.replace("-", "")
        return f"{EDGAR_BASE}/Archives/edgar/data/{int(cik)}/{acc_clean}/{accession}-index.htm"

    def _html_to_text(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return self._clean_text(text)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\x00", "", text)
        return text.strip()


async def fetch_sample_filings(output_dir: Path, tickers: list[str], filing_type: FilingType = "10-K") -> list[Path]:
    """Convenience function for CLI / scripts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    async with SECFetcher() as fetcher:
        for ticker in tickers:
            try:
                filings = await fetcher.get_filings(ticker, filing_type=filing_type, limit=3)
                for filing in filings:
                    fname = output_dir / f"{filing.doc_id}.txt"
                    fname.write_text(filing.text, encoding="utf-8")
                    saved.append(fname)
                    log.info("filing_saved", path=str(fname))
            except Exception as exc:
                log.error("fetch_failed", ticker=ticker, error=str(exc))

    return saved
