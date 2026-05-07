from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

from parcelsim.demand.base import ParcelDemand

if TYPE_CHECKING:
    from parcelsim.population.base import SyntheticPopulation

# USPS parcel generation factors U_i: weekly postal-service parcels per household by income.
# Source: USPS Household Diary Study 2020 (approximate values documented in Yang et al. 2024).
# These are USPS-only volumes; total across all carriers = U_i / usps_market_share.
PARCEL_GENERATION_FACTORS: dict[str, float] = {
    "lt35k":    0.50,   # < $35,000/yr
    "35k_65k":  0.72,   # $35,000–$64,999/yr
    "65k_100k": 0.95,   # $65,000–$99,999/yr
    "gt100k":   1.25,   # ≥ $100,000/yr
}

# Market shares (Pitney Bowes Parcel Shipping Index 2022, Yang et al. Table 1)
US_MARKET_SHARES: dict[str, float] = {
    "usps":   0.32,
    "ups":    0.25,
    "fedex":  0.20,
    "amazon": 0.23,
}

# Pickup volume thresholds differ from delivery — apply uniform scale
PICKUP_DELIVERY_RATIO = 0.08   # ~8% of delivery volume, per Yang et al. Table 1


class USPSDemandModel:
    """
    Parcel volume model from Yang, Landes & Chow (2024).

    V_ad = Σ_i (N_i × U_i × F) / (ω_p × d)

    where
      N_i  = number of households at income level i
      U_i  = USPS parcel generation factor (weekly parcels/hh by income)
      F    = volume increase factor (e-commerce growth adjustment)
      ω_p  = USPS postal market share (residential parcels)
      d    = delivery days per week

    The formula yields total residential daily parcels across ALL carriers.
    """

    def __init__(
        self,
        parcel_generation_factors: dict[str, float] | str = "builtin:usps_2020",
        volume_increase_factor: float = 1.114,
        usps_market_share: float = 0.32,
        delivery_days_per_week: int = 5,
        pickup_delivery_ratio: float = PICKUP_DELIVERY_RATIO,
    ) -> None:
        if isinstance(parcel_generation_factors, str):
            if parcel_generation_factors.startswith("builtin:"):
                self.pgf = PARCEL_GENERATION_FACTORS
            else:
                raise ValueError(f"Unknown builtin: {parcel_generation_factors}")
        else:
            self.pgf = parcel_generation_factors

        self.F = volume_increase_factor
        self.omega_p = usps_market_share
        self.d = delivery_days_per_week
        self.pickup_ratio = pickup_delivery_ratio

    def generate(self, population: "SyntheticPopulation") -> ParcelDemand:
        """Compute daily parcel demand at census tract (zone) level."""
        hh = population.households

        if "income_bracket" not in hh.columns:
            raise ValueError(
                "households must have an 'income_bracket' column. "
                "Use USCensusAdapter or assign income brackets manually."
            )

        # Count households per zone per income bracket
        hh_counts = (
            hh.groupby(["zone_id", "income_bracket"])
            .size()
            .reset_index(name="n_hh")
        )

        # Apply Yang et al. Eq.(1): V = N_i * U_i * F / (omega_p * d)
        hh_counts["U_i"] = hh_counts["income_bracket"].map(self.pgf).fillna(0.0)
        hh_counts["delivery_contrib"] = (
            hh_counts["n_hh"] * hh_counts["U_i"] * self.F / (self.omega_p * self.d)
        )

        zone_delivery = (
            hh_counts.groupby("zone_id")["delivery_contrib"]
            .sum()
            .reset_index()
            .rename(columns={"delivery_contrib": "n_delivery"})
        )
        zone_delivery["n_pickup"] = zone_delivery["n_delivery"] * self.pickup_ratio

        # Poisson assumption: Var = mean  →  std = sqrt(mean)
        zone_delivery["n_delivery_std"] = np.sqrt(zone_delivery["n_delivery"])
        zone_delivery["n_delivery_p05"] = np.maximum(
            0, zone_delivery["n_delivery"] - 1.645 * zone_delivery["n_delivery_std"]
        )
        zone_delivery["n_delivery_p95"] = (
            zone_delivery["n_delivery"] + 1.645 * zone_delivery["n_delivery_std"]
        )
        zone_delivery["n_pickup_std"] = np.sqrt(zone_delivery["n_pickup"])

        # Join zone geometry from city.zones
        city_zones = population.city.zones[["zone_id", "geometry", "area_km2",
                                             "centroid_x", "centroid_y"]]
        zone_demand = city_zones.merge(zone_delivery, on="zone_id", how="left")
        for col in ["n_delivery", "n_pickup",
                    "n_delivery_std", "n_delivery_p05", "n_delivery_p95", "n_pickup_std"]:
            zone_demand[col] = zone_demand[col].fillna(0.0)

        zone_demand = gpd.GeoDataFrame(zone_demand, geometry="geometry",
                                       crs=population.city.crs)

        total_delivery = float(zone_demand["n_delivery"].sum())
        total_pickup = float(zone_demand["n_pickup"].sum())

        return ParcelDemand(
            population=population,
            zone_demand=zone_demand,
            total_delivery=total_delivery,
            total_pickup=total_pickup,
            demand_model="usps",
            metadata={
                "F": self.F,
                "omega_p": self.omega_p,
                "d": self.d,
                "pgf": self.pgf,
                "n_zones_with_demand": int((zone_demand["n_delivery"] > 0).sum()),
            },
        )
