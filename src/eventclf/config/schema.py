from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    bounds: Optional[Tuple[float, float]] = None
    allow_nan_frac: float = 0.0


@dataclass(frozen=True)
class DatasetSchema:
    features: Tuple[FeatureSpec, ...]
    label: str
    weight: Optional[str] = None
    event_id: Optional[str] = "event"
    extra_cols: Tuple[str, ...] = ()

    def feature_names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @classmethod
    def from_feature_names(
        cls,
        feature_names: Tuple[str, ...] | list[str],
        label: str = "label",
        weight: Optional[str] = None,
        event_id: Optional[str] = "event",
        extra_cols: Tuple[str, ...] = (),
    ) -> "DatasetSchema":
        return cls(
            features=tuple(FeatureSpec(name) for name in feature_names),
            label=label,
            weight=weight,
            event_id=event_id,
            extra_cols=extra_cols,
        )


# ---- Example schema (MUST be defined at module level) ----
HMUMU_ZH2L_SCHEMA = DatasetSchema(
    features=(
        FeatureSpec("Muons_PT_Lead"),
        FeatureSpec("Muons_PT_Sub"),
        FeatureSpec("Event_VT_over_HT"),
        FeatureSpec("dR_mu0_mu1"),
        FeatureSpec("Jets_jetMultip"),
        FeatureSpec("Event_MET"),
        FeatureSpec("Muons_CosThetaStar"),
        FeatureSpec("Event_MET_Sig"),
        #FeatureSpec("DPHI_MET_DIMU"),
    ),
    label="label",          # we will create this in the script (0/1)
    weight=None,            # set to "weight" later if your ntuples have it
    event_id="event",# if missing, we’ll fallback to index
    extra_cols=(),
)
