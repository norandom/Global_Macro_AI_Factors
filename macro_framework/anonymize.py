"""Asset anonymization — hides tickers/names from the LLM to block training-data lookahead.

Mapping is fixed and consistent within a run so the agent's letter references can be
translated back to real tickers after each call. Category hints are preserved because
the agent needs *some* basis for causal reasoning (you can't reason about an asset
class without knowing it's an equity vs a commodity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

DEFAULT_MAPPING: dict[str, tuple[str, str]] = {
    "SWDA.L": ("Asset_A", "world_equity"),
    "XLK":    ("Asset_B", "tech_sector"),
    "IAU":    ("Asset_C", "gold_commodity"),
    "BIL":    ("Asset_D", "short_treasury_cash"),
}


@dataclass(frozen=True)
class AssetMap:
    real_to_pseudo: dict[str, str]
    pseudo_to_real: dict[str, str]
    categories:    dict[str, str] = field(default_factory=dict)  # keyed by pseudo name

    @classmethod
    def default(cls) -> "AssetMap":
        real_to_pseudo = {real: p for real, (p, _) in DEFAULT_MAPPING.items()}
        pseudo_to_real = {p: real for real, p in real_to_pseudo.items()}
        categories = {p: cat for _, (p, cat) in DEFAULT_MAPPING.items()}
        return cls(real_to_pseudo=real_to_pseudo, pseudo_to_real=pseudo_to_real, categories=categories)

    def anonymize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.real_to_pseudo)

    def deanonymize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns=self.pseudo_to_real)

    def anonymize_series(self, s: pd.Series) -> pd.Series:
        return s.rename(index=self.real_to_pseudo)

    def deanonymize_series(self, s: pd.Series) -> pd.Series:
        return s.rename(index=self.pseudo_to_real)

    def pseudo_assets(self) -> list[dict[str, str]]:
        """Agent-facing asset descriptor list — letters + category only, no ticker."""
        return [{"id": p, "category": self.categories[p]} for p in self.real_to_pseudo.values()]
