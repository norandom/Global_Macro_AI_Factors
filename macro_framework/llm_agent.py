"""Track A — zero-temp DSPy agent emitting Black-Litterman views from anonymized macro state.

Wiring is lazy (DSPy and diskcache imported inside the class) so `import macro_framework`
stays cheap. The agent never sees a date, a year, or a real ticker.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .anonymize import AssetMap

PROMPT_VERSION = "v1"
OPENROUTER_MODEL = "openrouter/anthropic/claude-sonnet-4.6"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / ".llm_cache"

AGENT_INSTRUCTIONS = """You are a macroeconomic game theorist. You are given the current
macro state (z-scored against a rolling 5-year window) and an anonymized list of four
assets identified only by letter and category.

Your job:
 1. Interpret the macro state. Identify any causal contradictions or tail regimes
    (e.g., high inflation + inverted curve + widening credit spreads = stagflation-risk).
 2. Emit concrete Black-Litterman **view triples** that tilt a portfolio against the
    dominant risk. A view has one "long" asset and optionally one "short" asset, an
    expected annualized excess return (decimal, e.g. 0.05 = +5%/yr), and a confidence
    0..1 (0 = no information, 1 = certainty — practically top out at ~0.6).
 3. Keep views few (1-3). Prefer higher-quality views over quantity.
 4. Never reference calendar dates, years, or real tickers. You do not know what
    year it is. Reason only from the numeric state in front of you.

Output JSON array of views only. Use the asset letters exactly as given (e.g. "Asset_A").
"""


@dataclass
class MacroView:
    asset_long: str                       # pseudo (e.g., "Asset_A")
    asset_short: str | None               # pseudo or None
    expected_excess_annualized: float     # decimal (0.05 = 5%/yr)
    confidence: float                     # 0..1
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_long": self.asset_long,
            "asset_short": self.asset_short,
            "expected_excess_annualized": self.expected_excess_annualized,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


def _cache_key(macro_state: dict[str, float], assets: list[dict[str, str]]) -> str:
    payload = {"v": PROMPT_VERSION, "m": macro_state, "a": assets}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class LlmMacroAgent:
    def __init__(
        self,
        asset_map: AssetMap | None = None,
        model: str = OPENROUTER_MODEL,
        api_base: str = OPENROUTER_API_BASE,
        api_key_env: str = "OPENROUTER_KEY",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cache_dir: str | Path = CACHE_DIR,
    ):
        self.asset_map = asset_map or AssetMap.default()
        self.model = model
        self.api_base = api_base
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir)
        self._lm = None
        self._predict = None
        self._cache = None

    # ---- lazy init --------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._predict is not None:
            return
        import diskcache
        import dspy
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} not set in environment / .env")

        self._lm = dspy.LM(
            model=self.model,
            api_key=api_key,
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        dspy.settings.configure(lm=self._lm)

        class MacroViewSignature(dspy.Signature):
            __doc__ = AGENT_INSTRUCTIONS
            macro_state: dict = dspy.InputField(
                desc="z-scored macro state: cpi_yoy_z, t10y2y_z, hy_oas_z (float each)"
            )
            assets:      list = dspy.InputField(
                desc="anonymized assets [{id, category, trailing_12m_return, trailing_vol_ann}]"
            )
            reasoning:   str  = dspy.OutputField(
                desc="2-4 sentence causal analysis of the macro state"
            )
            views_json:  str  = dspy.OutputField(
                desc='JSON array: [{"asset_long":"Asset_X","asset_short":"Asset_Y or null",'
                     '"expected_excess_annualized":float,"confidence":0..1,"rationale":str}]'
            )

        self._predict = dspy.Predict(MacroViewSignature)
        self._cache = diskcache.Cache(str(self.cache_dir))

    # ---- core API ---------------------------------------------------------

    def views_for_state(
        self,
        macro_state: dict[str, float],
        asset_snapshot: list[dict[str, Any]],
    ) -> tuple[list[MacroView], str]:
        """Request views. Cached on (rounded_state, assets, prompt_version)."""
        self._ensure_ready()
        macro_rounded = {k: round(float(v), 2) for k, v in macro_state.items()}
        assets_clean = [
            {k: (round(float(v), 3) if isinstance(v, (int, float)) else v) for k, v in a.items()}
            for a in asset_snapshot
        ]
        key = _cache_key(macro_rounded, assets_clean)

        cached = self._cache.get(key)
        if cached is not None:
            return [MacroView(**v) for v in cached["views"]], cached["reasoning"]

        out = self._predict(macro_state=macro_rounded, assets=assets_clean)
        reasoning = str(getattr(out, "reasoning", "") or "")
        views = self._parse_views(getattr(out, "views_json", "") or "[]")
        self._cache.set(key, {"views": [v.to_dict() for v in views], "reasoning": reasoning})
        return views, reasoning

    @staticmethod
    def _parse_views(raw: str) -> list[MacroView]:
        s = raw.strip()
        # strip markdown fences if present
        if s.startswith("```"):
            s = s.strip("`").lstrip()
            if s.startswith("json"):
                s = s[4:].lstrip()
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        out: list[MacroView] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                out.append(MacroView(
                    asset_long=str(item["asset_long"]),
                    asset_short=(str(item["asset_short"]) if item.get("asset_short") else None),
                    expected_excess_annualized=float(item["expected_excess_annualized"]),
                    confidence=float(item["confidence"]),
                    rationale=str(item.get("rationale", "")),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    # ---- view translation -------------------------------------------------

    def views_to_bl(
        self,
        views: list[MacroView],
        real_symbols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[None, None]:
        """Convert LLM views (pseudo letters) into BL (P, Q) keyed by real tickers.

        Q is confidence-gated: effective_Q = expected_excess_annualized * confidence / 252.
        Lower confidence ⇒ proportionally weaker tilt via Q magnitude.
        """
        pseudo_to_real = self.asset_map.pseudo_to_real
        rows_P, rows_Q = [], []
        for v in views:
            long_real = pseudo_to_real.get(v.asset_long)
            short_real = pseudo_to_real.get(v.asset_short) if v.asset_short else None
            if long_real is None or long_real not in real_symbols:
                continue
            if v.asset_short and (short_real is None or short_real not in real_symbols):
                continue
            p = np.zeros(len(real_symbols))
            p[real_symbols.index(long_real)] = 1.0
            if short_real:
                p[real_symbols.index(short_real)] = -1.0
            conf = max(0.0, min(1.0, v.confidence))
            q = (v.expected_excess_annualized * conf) / 252.0
            rows_P.append(p)
            rows_Q.append([q])
        if not rows_P:
            return None, None
        P = pd.DataFrame(rows_P, columns=real_symbols)
        Q = pd.DataFrame(rows_Q)
        return P, Q
