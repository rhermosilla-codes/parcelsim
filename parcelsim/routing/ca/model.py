from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from parcelsim.operators.assignment import OperatorAssignment
    from parcelsim.city import City
    from parcelsim.scenario.base import Scenario

# Calibrated k coefficients from Yang et al. (2024) Table 3, fitted on NYC
# k reflects road network density: denser network → lower k (shorter local tours)
NYC_K_COEFFICIENTS: dict[str, float] = {
    "MN": 0.708,    # Manhattan — densest grid
    "BX": 0.894,    # Bronx
    "BK": 0.856,    # Brooklyn
    "QN": 0.856,    # Queens (combined with Brooklyn in paper)
    "SI": 0.993,    # Staten Island — lowest density
}

DEFAULT_K = 0.85        # Generic fallback for unknown cities (between MN and BK/QN)
TRUCK_CAPACITY = 300    # parcels per truck per day (CVRP calibration assumption)


@dataclass
class CAResult:
    """Output of the Continuous Approximation router."""

    operator_id: str
    depot_results: pd.DataFrame   # columns: depot_id, n_zones, n_stops, area_km2,
                                  #          r_km, m_trucks, vkt_km
    vkt_total_km: float
    n_trucks_total: int
    n_stops_total: int
    scenario_name: str = "baseline"
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"CAResult [{self.operator_id}] scenario={self.scenario_name}\n"
            f"  VKT total:    {self.vkt_total_km:>12,.1f} km/day\n"
            f"  Trucks:       {self.n_trucks_total:>12,}\n"
            f"  Stops:        {self.n_stops_total:>12,}"
        )


@dataclass
class CARoutingResult:
    """Aggregated CA result across all operators."""

    operator_results: list[CAResult]
    scenario_name: str = "baseline"

    @property
    def vkt_total_km(self) -> float:
        return sum(r.vkt_total_km for r in self.operator_results)

    @property
    def n_trucks_total(self) -> int:
        return sum(r.n_trucks_total for r in self.operator_results)

    def by_operator(self) -> pd.DataFrame:
        rows = [
            {
                "operator_id": r.operator_id,
                "vkt_km": r.vkt_total_km,
                "n_trucks": r.n_trucks_total,
                "n_stops": r.n_stops_total,
            }
            for r in self.operator_results
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [f"CARoutingResult  scenario={self.scenario_name}",
                 f"  Total VKT:  {self.vkt_total_km:>12,.1f} km/day",
                 f"  Total trucks: {self.n_trucks_total:>10,}",
                 "  By operator:"]
        for r in self.operator_results:
            lines.append(
                f"    {r.operator_id:10s}  VKT={r.vkt_total_km:>10,.1f} km  "
                f"trucks={r.n_trucks_total:>5,}"
            )
        return "\n".join(lines)


class CARouter:
    """
    Continuous Approximation router implementing Daganzo (1984) / Yang et al. (2024).

    Formula (Figliozzi 2008 correction for multi-truck):
        V = 2·r·m + k·(n - m)·√(A/n)

    where:
        V  = vehicle-km-traveled for one depot service area
        r  = depot-to-centroid distance (km)
        m  = number of trucks
        n  = number of service stops
        A  = service area (km²)
        k  = network geometry coefficient (calibrated per region)
    """

    def __init__(
        self,
        k_coefficients: dict[str, float] | None = None,
        default_k: float = DEFAULT_K,
        truck_capacity: int = TRUCK_CAPACITY,
        stops_per_parcel: float = 1.0,
    ) -> None:
        self.k_map = k_coefficients or {}
        self.default_k = default_k
        self.truck_capacity = truck_capacity
        self.stops_per_parcel = stops_per_parcel

    def solve(
        self,
        assignment: "OperatorAssignment",
        city: "City",
        scenario: "Scenario | None" = None,
    ) -> CARoutingResult:
        demand_factor = 1.0
        peak_factor = 1.0
        if scenario is not None:
            demand_factor = scenario.demand_factor
            peak_factor = scenario.peak_factor

        operator_results = []
        for op in assignment.registry.operators:
            result = self._solve_operator(
                op.operator_id, assignment, city, demand_factor * peak_factor
            )
            if scenario is not None:
                result.scenario_name = scenario.name
            operator_results.append(result)

        return CARoutingResult(
            operator_results=operator_results,
            scenario_name=scenario.name if scenario else "baseline",
        )

    def _solve_operator(
        self,
        operator_id: str,
        assignment: "OperatorAssignment",
        city: "City",
        volume_factor: float = 1.0,
    ) -> CAResult:
        zones = assignment.for_operator(operator_id)
        zones = zones[zones["n_delivery"] > 0].copy()
        zones["n_delivery"] = zones["n_delivery"] * volume_factor

        if zones.empty:
            return CAResult(
                operator_id=operator_id,
                depot_results=pd.DataFrame(),
                vkt_total_km=0.0,
                n_trucks_total=0,
                n_stops_total=0,
            )

        depot_rows = []
        for depot_id, group in zones.groupby("depot_id"):
            row = self._solve_depot(depot_id, group, city)
            depot_rows.append(row)

        depot_df = pd.DataFrame(depot_rows)
        return CAResult(
            operator_id=operator_id,
            depot_results=depot_df,
            vkt_total_km=float(depot_df["vkt_km"].sum()),
            n_trucks_total=int(depot_df["m_trucks"].sum()),
            n_stops_total=int(depot_df["n_stops"].sum()),
        )

    def _solve_depot(
        self,
        depot_id: str,
        zones: gpd.GeoDataFrame,
        city: "City",
    ) -> dict:
        # Service area metrics
        area_km2 = float(zones["area_km2"].sum())
        total_parcels = float(zones["n_delivery"].sum())

        # Stops: each zone contributes proportional to its parcel volume
        # Threshold-based aggregation is approximated here by treating each
        # zone above 1 parcel as a stop (adequate for CA-level analysis)
        n_stops = int(np.round(total_parcels * self.stops_per_parcel))
        n_stops = max(n_stops, 1)

        # Number of trucks (round n/capacity, minimum 1)
        m = max(1, int(np.round(n_stops / self.truck_capacity)))

        # r = distance from depot to service area centroid
        # Use weighted centroid of all zones (weighted by n_delivery)
        w = zones["n_delivery"].values
        cx = zones["centroid_x"].values
        cy = zones["centroid_y"].values
        if w.sum() > 0:
            service_cx = float(np.average(cx, weights=w))
            service_cy = float(np.average(cy, weights=w))
        else:
            service_cx = float(cx.mean())
            service_cy = float(cy.mean())

        # Depot location from assignment distance column
        r_km = float(zones["dist_km"].mean())

        # k coefficient — look up by zone region code (first 2 chars of zone_id)
        k = self._get_k(zones, city)

        # Daganzo (1984) formula with Figliozzi (2008) multi-truck correction:
        # V = 2·r·m + k·(n - m)·√(A/n)
        if n_stops > m and area_km2 > 0:
            local_tour = k * (n_stops - m) * np.sqrt(area_km2 / n_stops)
        else:
            local_tour = 0.0
        vkt_km = 2.0 * r_km * m + local_tour

        return {
            "depot_id": depot_id,
            "n_zones": len(zones),
            "n_stops": n_stops,
            "area_km2": area_km2,
            "r_km": r_km,
            "k": k,
            "m_trucks": m,
            "vkt_km": vkt_km,
        }

    def _get_k(self, zones: gpd.GeoDataFrame, city: "City") -> float:
        if self.k_map:
            return self.default_k

        # Attempt to infer NYC borough from zone_id prefix (11-digit GEOID)
        # FIPS county codes: 061=MN, 005=BX, 047=BK, 081=QN, 085=SI
        county_to_k = {"061": 0.708, "005": 0.894, "047": 0.856, "081": 0.856, "085": 0.993}
        sample_zone = zones["zone_id"].iloc[0] if len(zones) > 0 else ""
        # GEOID format: STATE(2) + COUNTY(3) + TRACT(6)
        if len(sample_zone) >= 5:
            county = sample_zone[2:5]
            if county in county_to_k:
                return county_to_k[county]

        return self.default_k
