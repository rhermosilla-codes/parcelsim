from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd

if TYPE_CHECKING:
    from parcelsim.population.base import SyntheticPopulation


@dataclass
class ParcelDemand:
    """
    Daily parcel demand at zone level, ready for operator assignment.

    zone_demand GeoDataFrame required columns:
        zone_id         str
        geometry        Polygon / Point centroid
        n_delivery      float   average daily delivery parcels
        n_pickup        float   average daily pickup parcels
        area_km2        float   zone area in km²
        centroid_x      float   projected X of zone centroid
        centroid_y      float   projected Y of zone centroid

    demand_factor and peak_factor are scenario multipliers applied on top.
    """

    population: "SyntheticPopulation"
    zone_demand: gpd.GeoDataFrame
    total_delivery: float
    total_pickup: float
    demand_model: str
    demand_factor: float = 1.0
    peak_factor: float = 1.0
    metadata: dict = field(default_factory=dict)

    @property
    def effective_delivery(self) -> float:
        return self.total_delivery * self.demand_factor * self.peak_factor

    @property
    def effective_pickup(self) -> float:
        return self.total_pickup * self.demand_factor * self.peak_factor

    def apply_factor(self, demand_factor: float = 1.0, peak_factor: float = 1.0) -> "ParcelDemand":
        """Return a new ParcelDemand with updated scenario multipliers (non-destructive)."""
        import copy
        d = copy.copy(self)
        d.demand_factor = demand_factor
        d.peak_factor = peak_factor
        return d

    def summary(self) -> str:
        return (
            f"ParcelDemand [{self.demand_model}]\n"
            f"  Total delivery (base): {self.total_delivery:,.0f} parcels/day\n"
            f"  Total pickup  (base): {self.total_pickup:,.0f} parcels/day\n"
            f"  Effective delivery:   {self.effective_delivery:,.0f} parcels/day\n"
            f"  Zones with demand:    {len(self.zone_demand):,}"
        )
