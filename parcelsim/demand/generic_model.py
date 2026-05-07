from __future__ import annotations

from pathlib import Path

import numpy as np
import geopandas as gpd
import yaml

from parcelsim.demand.base import ParcelDemand
from parcelsim.population.base import SyntheticPopulation

_PARAMS_FILE = Path(__file__).parent / "builtin" / "country_params.yaml"


def _load_params() -> dict:
    with open(_PARAMS_FILE) as f:
        return yaml.safe_load(f)


class AggregateDemandModel:
    """
    Country-agnostic aggregate parcel demand model.

    Uses a single household rate (no income stratification required).
    Compatible with any population adapter — including WorldPop.

    Parameters can be supplied directly or loaded from the built-in country
    registry via `AggregateDemandModel.from_country(iso)`.

    Parameters
    ----------
    annual_parcels_per_hh : float
        Average e-commerce parcels delivered per household per year.
    home_delivery_fraction : float
        Share of parcels delivered to home (vs. pickup point / locker).
    delivery_days_per_year : int
        Working delivery days per year.
    demand_factor : float
        Volume growth multiplier relative to the source year (default 1.0).
    seed : int
        Random seed for Poisson draws.
    """

    def __init__(
        self,
        annual_parcels_per_hh: float,
        home_delivery_fraction: float,
        delivery_days_per_year: int = 250,
        demand_factor: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.annual_parcels_per_hh = annual_parcels_per_hh
        self.home_delivery_fraction = home_delivery_fraction
        self.delivery_days_per_year = delivery_days_per_year
        self.demand_factor = demand_factor
        self.seed = seed

    @classmethod
    def from_country(cls, iso: str, demand_factor: float = 1.0, seed: int = 42) -> "AggregateDemandModel":
        """
        Load parameters from the built-in country registry.

        Parameters
        ----------
        iso : str
            ISO 3166-1 alpha-2 country code (e.g. "DE", "GB", "BR").
        demand_factor : float
            Growth multiplier relative to the survey year (default 1.0).

        Raises
        ------
        KeyError
            If the country is not in the registry. Call
            `AggregateDemandModel.available_countries()` to list supported codes.
        """
        params = _load_params()
        iso = iso.upper()
        if iso not in params:
            available = sorted(params.keys())
            raise KeyError(
                f"Country '{iso}' not in registry. "
                f"Available: {available}\n"
                f"To add a new country edit: {_PARAMS_FILE}"
            )
        p = params[iso]
        return cls(
            annual_parcels_per_hh=p["annual_parcels_per_hh"],
            home_delivery_fraction=p["home_delivery_fraction"],
            delivery_days_per_year=p["delivery_days_per_year"],
            demand_factor=demand_factor,
            seed=seed,
        )

    @classmethod
    def available_countries(cls) -> dict[str, dict]:
        """Return the full country registry as a dict keyed by ISO code."""
        return _load_params()

    @property
    def daily_rate_per_hh(self) -> float:
        return (
            self.annual_parcels_per_hh
            * self.home_delivery_fraction
            * self.demand_factor
            / self.delivery_days_per_year
        )

    def generate(self, population: SyntheticPopulation) -> ParcelDemand:
        """Generate zone-level parcel demand with mean and std (Poisson model)."""
        rng = np.random.default_rng(self.seed)
        city = population.city
        zones = city.zones.copy()

        hh_per_zone = (
            population.households.groupby("zone_id")["n_households"].sum()
            if "n_households" in population.households.columns
            else population.households.groupby("zone_id").size()
        )

        rows = []
        for _, zone in zones.iterrows():
            zid = zone["zone_id"]
            n_hh = float(hh_per_zone.get(zid, 0))
            if n_hh == 0:
                continue
            mu = n_hh * self.daily_rate_per_hh
            std = float(np.sqrt(mu))
            rows.append({
                "zone_id": zid,
                "geometry": zone.geometry,
                "centroid_x": zone.get("centroid_x", zone.geometry.centroid.x),
                "centroid_y": zone.get("centroid_y", zone.geometry.centroid.y),
                "area_km2": zone.get("area_km2", zone.geometry.area / 1e6),
                "n_households": n_hh,
                "n_delivery": float(rng.poisson(mu)),
                "n_delivery_std": std,
                "n_delivery_p05": max(0.0, mu - 1.645 * std),
                "n_delivery_p95": mu + 1.645 * std,
                "n_pickup": 0.0,
                "n_pickup_std": 0.0,
            })

        zone_demand = gpd.GeoDataFrame(rows, geometry="geometry", crs=city.crs)
        total_delivery = float(zone_demand["n_delivery"].sum())

        return ParcelDemand(
            population=population,
            zone_demand=zone_demand,
            total_delivery=total_delivery,
            total_pickup=0.0,
            demand_model=f"aggregate",
            metadata={
                "annual_parcels_per_hh": self.annual_parcels_per_hh,
                "home_delivery_fraction": self.home_delivery_fraction,
                "delivery_days_per_year": self.delivery_days_per_year,
                "demand_factor": self.demand_factor,
                "daily_rate_per_hh": self.daily_rate_per_hh,
            },
        )
