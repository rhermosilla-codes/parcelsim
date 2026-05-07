from __future__ import annotations

import numpy as np
import geopandas as gpd

from parcelsim.demand.base import ParcelDemand
from parcelsim.population.base import SyntheticPopulation

# Annual remote purchases per household (Gardrat 2019, Hörl et al. 2025)
ANNUAL_PURCHASES_PER_HH = 14.0

# Share of purchases delivered at home vs. pickup point (Gardrat 2019 survey)
HOME_DELIVERY_FRACTION = 0.47

# Demand growth factors relative to 2019 baseline (Hörl et al. Table 5)
DEMAND_FACTORS = {2019: 1.00, 2024: 1.35, 2030: 2.00}


class FranceDemandModel:
    """
    Simplified parcel demand model for France, based on Hörl, Briand & Puchinger (2025).

    Uses aggregate household purchase rates from Gardrat (2019) survey without
    full IPF fitting — appropriate when socio-demographic breakdowns are unavailable
    (e.g. when using WorldPopAdapter).

    Formula (Hörl et al. Eq. 3, simplified):
        μ_hh = annual_purchases × demand_factor × home_delivery_fraction / delivery_days

    Parameters
    ----------
    annual_purchases_per_hh : float
        Total annual remote purchases per household (default 14, Gardrat 2019).
    home_delivery_fraction : float
        Share of purchases delivered at home (default 0.47, Gardrat 2019).
    demand_factor : float
        Volume growth factor relative to 2019. Use DEMAND_FACTORS for standard years.
    delivery_days_per_year : int
        Working delivery days per year (default 260, France).
    seed : int
        Random seed for Poisson draws.
    """

    def __init__(
        self,
        annual_purchases_per_hh: float = ANNUAL_PURCHASES_PER_HH,
        home_delivery_fraction: float = HOME_DELIVERY_FRACTION,
        demand_factor: float = DEMAND_FACTORS[2024],
        delivery_days_per_year: int = 260,
        seed: int = 42,
    ) -> None:
        self.annual_purchases_per_hh = annual_purchases_per_hh
        self.home_delivery_fraction = home_delivery_fraction
        self.demand_factor = demand_factor
        self.delivery_days_per_year = delivery_days_per_year
        self.seed = seed

    @property
    def daily_rate_per_hh(self) -> float:
        """Expected daily parcels per household."""
        return (
            self.annual_purchases_per_hh
            * self.demand_factor
            * self.home_delivery_fraction
            / self.delivery_days_per_year
        )

    def generate(self, population: SyntheticPopulation) -> ParcelDemand:
        """
        Generate zone-level parcel demand.

        Returns
        -------
        ParcelDemand
            Zone demand with n_delivery per zone, n_pickup = 0 (France model
            does not model returns/pickups separately).
        """
        rng = np.random.default_rng(self.seed)
        city = population.city
        zones = city.zones.copy()

        # Aggregate household count per zone from population
        hh_per_zone = (
            population.households
            .groupby("zone_id")["n_households"]
            .sum()
            if "n_households" in population.households.columns
            else population.households.groupby("zone_id").size()
        )

        zone_demand_rows = []
        total_delivery = 0.0

        for _, zone in zones.iterrows():
            zid = zone["zone_id"]
            n_hh = float(hh_per_zone.get(zid, 0))
            if n_hh == 0:
                continue

            # Expected daily deliveries: Poisson mean
            mu = n_hh * self.daily_rate_per_hh
            # Poisson draw for realistic integer count
            n_delivery = float(rng.poisson(mu))
            total_delivery += n_delivery

            std = float(np.sqrt(mu))   # Poisson: std = sqrt(mean)
            zone_demand_rows.append({
                "zone_id": zid,
                "geometry": zone.geometry,
                "centroid_x": zone.get("centroid_x", zone.geometry.centroid.x),
                "centroid_y": zone.get("centroid_y", zone.geometry.centroid.y),
                "area_km2": zone.get("area_km2", zone.geometry.area / 1e6),
                "n_households": n_hh,
                "n_delivery": n_delivery,
                "n_delivery_std": std,
                "n_delivery_p05": max(0.0, mu - 1.645 * std),
                "n_delivery_p95": mu + 1.645 * std,
                "n_pickup": 0.0,
                "n_pickup_std": 0.0,
            })

        zone_demand = gpd.GeoDataFrame(zone_demand_rows, geometry="geometry", crs=city.crs)

        return ParcelDemand(
            population=population,
            zone_demand=zone_demand,
            total_delivery=total_delivery,
            total_pickup=0.0,
            demand_model="france_gardrat",
        )
