from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd

if TYPE_CHECKING:
    from parcelsim.city import City


@dataclass
class SyntheticPopulation:
    """
    Households with location and sociodemographic attributes.

    households GeoDataFrame required columns:
        household_id  str
        zone_id       str
        geometry      Point   (building location or zone centroid)
        n_persons     int
        income_bracket str   e.g. "lt35k", "35k_65k", "65k_100k", "gt100k"

    Optional columns (richer models):
        age_ref       int    age of reference person
        spc_category  str    socio-professional category (France)
    """

    city: "City"
    households: gpd.GeoDataFrame
    source_adapter: str
    year: int
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {"household_id", "zone_id", "geometry", "n_persons", "income_bracket"}
        missing = required - set(self.households.columns)
        if missing:
            raise ValueError(f"households GeoDataFrame missing columns: {missing}")

    @property
    def n_households(self) -> int:
        return len(self.households)

    @property
    def n_persons(self) -> int:
        return int(self.households["n_persons"].sum())

    def summary(self) -> str:
        lines = [
            f"SyntheticPopulation [{self.source_adapter}, {self.year}]",
            f"  City:        {self.city.name}",
            f"  Households:  {self.n_households:,}",
            f"  Persons:     {self.n_persons:,}",
            f"  Zones:       {self.households['zone_id'].nunique():,}",
        ]
        if "income_bracket" in self.households.columns:
            dist = self.households["income_bracket"].value_counts()
            lines.append("  Income dist: " + ", ".join(f"{k}={v:,}" for k, v in dist.items()))
        return "\n".join(lines)
