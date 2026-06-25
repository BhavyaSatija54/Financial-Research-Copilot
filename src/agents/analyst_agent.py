"""
Analyst Agent — financial synthesis using LangChain + GPT-4o.

Responsibilities:
  - Build a context window from reranked chunks
  - Classify query intent (risk, financials, guidance, macro, etc.)
  - Generate a structured financial analysis response
  - Enforce source faithfulness (no hallucination)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.retrieval.reranker import RankedResult
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.metrics import track_latency

log = get_logger(__name__)

# ── Intent classification patterns ───────────────────────────────────
_INTENT_PATTERNS: dict[str, list[str]] = {
    "risk_factors": ["risk", "risk factor", "threat", "headwind", "exposure"],
    "financial_metrics": ["revenue", "eps", "earnings", "profit", "margin", "ebitda", "net income", "cash flow"],
    "guidance": ["guidance", "outlook", "forecast", "projection", "next quarter", "full year"],
    "macro": ["interest rate", "inflation", "gdp", "federal reserve", "macro", "economic"],
    "management": ["ceo", "management", "leadership", "strategy", "acquisition", "merger"],
    "comparison": ["compare", "versus", "vs", "benchmark", "peer", "industry"],
}

_SYSTEM_PROMPT = """You are a senior financial analyst specialising in SEC filings, earnings call transcripts,
and macroeconomic research. Your task is to answer the user's financial research question
using ONLY the provided document excerpts as your source of truth.

Guidelines:
- Be precise and use exact figures when they appear in the source material.
- Structure your response with clear sections when the answer is multi-faceted.
- If the source material does not contain enough information to answer definitively, say so explicitly.
- Do NOT speculate or use knowledge beyond what is provided in the context.
- When referencing data, indicate which document/filing it came from.
- For numerical comparisons, use tables in markdown format where appropriate.
- Flag any conflicting information across documents.

Response format:
1. Direct answer to the question (2–3 sentences max)
2. Supporting analysis with key data points
3. Important caveats or limitations of the available data
"""


@dataclass
class AnalysisResult:
    answer: str
    intent: str
    token_count: int


class AnalystAgent:
    """LangChain-powered financial analyst."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.llm = ChatOpenAI(
            model=cfg.openai_model,
            temperature=cfg.openai_temperature,
            max_tokens=cfg.openai_max_tokens,
            api_key=cfg.openai_api_key_str,
            streaming=False,
        )
        self._max_context_tokens = 12_000

    async def analyse(
        self,
        question: str,
        chunks: list[RankedResult],
    ) -> AnalysisResult:
        """
        Synthesise an answer from ranked chunks.

        Parameters
        ----------
        question : str
        chunks : list[RankedResult]
            Ordered by rerank score (best first).

        Returns
        -------
        AnalysisResult
        """
        intent = self._classify_intent(question)
        context = self._build_context(chunks)
        prompt = self._build_prompt(question, context, intent)

        log.info(
            "analyst_llm_call",
            intent=intent,
            context_chunks=len(chunks),
            prompt_chars=len(prompt),
        )

        with track_latency("llm_synthesis", intent=intent):
            messages = [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = await self.llm.ainvoke(messages)

        answer = response.content if isinstance(response.content, str) else str(response.content)
        token_count = response.usage_metadata.get("total_tokens", 0) if response.usage_metadata else 0

        log.info("analyst_done", intent=intent, answer_chars=len(answer), tokens=token_count)
        return AnalysisResult(answer=answer, intent=intent, token_count=token_count)

    # ── Private ───────────────────────────────────────────────────────

    def _classify_intent(self, question: str) -> str:
        q = question.lower()
        for intent, keywords in _INTENT_PATTERNS.items():
            if any(kw in q for kw in keywords):
                return intent
        return "general"

    def _build_context(self, chunks: list[RankedResult]) -> str:
        """Build a token-bounded context string from ranked chunks."""
        parts: list[str] = []
        total_chars = 0
        # Approximate: 1 token ≈ 4 chars; budget ~12k tokens → ~48k chars
        char_budget = self._max_context_tokens * 4

        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.metadata
            header = (
                f"[Source {i}] "
                f"{meta.get('company', meta.get('ticker', 'Unknown'))} | "
                f"{meta.get('filing_type', meta.get('doc_type', 'Document'))} | "
                f"Period: {meta.get('period', meta.get('filed_date', 'N/A'))}"
            )
            block = f"{header}\n{chunk.text}\n"

            if total_chars + len(block) > char_budget:
                log.debug("context_budget_reached", included_chunks=i - 1)
                break

            parts.append(block)
            total_chars += len(block)

        return "\n---\n".join(parts)

    def _build_prompt(self, question: str, context: str, intent: str) -> str:
        return (
            f"## Research Question\n{question}\n\n"
            f"## Query Intent\n{intent.replace('_', ' ').title()}\n\n"
            f"## Source Documents\n\n{context}\n\n"
            "## Your Analysis\n"
            "Please answer the research question based solely on the source documents above."
        )
