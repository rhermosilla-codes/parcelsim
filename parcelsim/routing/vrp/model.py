from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from parcelsim.routing.vrp.result import VRPDepotResult, VRPOperatorResult, VRPResult

if TYPE_CHECKING:
    from parcelsim.city import City
    from parcelsim.operators.assignment import OperatorAssignment
    from parcelsim.scenario.base import Scenario


@dataclass
class VehicleType:
    """Vehicle type specification (Hörl et al. 2025 Table 4)."""

    name: str
    capacity_parcels: int       # parcels per vehicle
    monthly_cost_eur: float     # EUR/month
    emission_g_co2_per_km: float
    fuel_l_per_100km: float = 0.0     # ICV only
    elec_wh_per_km: float = 0.0       # BEV only
    is_electric: bool = False

    @property
    def daily_fixed_cost_eur(self) -> float:
        """Monthly cost / 22 operating days."""
        return self.monthly_cost_eur / 22.0


# Hörl et al. (2025) Table 4 — France 2024 baseline vehicle types
VEHICLE_TYPES: dict[str, VehicleType] = {
    "small_icv":  VehicleType("Small ICV",   33,  210, 130.0, fuel_l_per_100km=5.0),
    "medium_icv": VehicleType("Medium ICV",  50,  260, 160.0, fuel_l_per_100km=6.0),
    "large_icv":  VehicleType("Large ICV",  100,  370, 215.0, fuel_l_per_100km=8.0),
    "small_bev":  VehicleType("Small BEV",   33,  260,  14.4, elec_wh_per_km=160.0, is_electric=True),
    "medium_bev": VehicleType("Medium BEV",  50,  400,  18.0, elec_wh_per_km=200.0, is_electric=True),
    "large_bev":  VehicleType("Large BEV",  100,  800,  27.0, elec_wh_per_km=300.0, is_electric=True),
}

# Driver salary (Hörl et al. 2025 Section 3.5.1)
DRIVER_DAILY_SALARY_EUR = 93.0

# Fuel price 2024 (EUR/L diesel)
FUEL_PRICE_EUR_PER_L = 1.80

# Electricity price 2024 (EUR/kWh)
ELEC_PRICE_EUR_PER_KWH = 0.2756


class VRPRouter:
    """
    Zone-level CVRP router using Google OR-Tools.

    Solves one capacitated vehicle routing problem per depot-operator pair.
    Delivery stops = zone centroids assigned to that depot.
    Vehicle demand = parcels to deliver in each zone.

    This is a planning-level approximation — for street-level routing use
    VROOM with individual household locations (Hörl et al. 2025).

    Parameters
    ----------
    vehicle_type : str
        Key from VEHICLE_TYPES dict (default "medium_icv").
    max_vehicles : int
        Maximum vehicles available per depot (default 40, Hörl et al.).
    time_limit_seconds : int
        OR-Tools solver time limit per depot instance.
    stops_per_parcel : float
        Delivery stops per parcel (default 1 stop = 1 parcel).
    fuel_price : float
        EUR per litre (ICV). Default = 1.80 (France 2024).
    elec_price : float
        EUR per kWh (BEV). Default = 0.2756 (France 2024).
    """

    def __init__(
        self,
        vehicle_type: str = "medium_icv",
        max_vehicles: int = 40,
        time_limit_seconds: int = 30,
        stops_per_parcel: float = 1.0,
        fuel_price: float = FUEL_PRICE_EUR_PER_L,
        elec_price: float = ELEC_PRICE_EUR_PER_KWH,
    ) -> None:
        if vehicle_type not in VEHICLE_TYPES:
            raise ValueError(f"Unknown vehicle_type '{vehicle_type}'. "
                             f"Available: {list(VEHICLE_TYPES)}")
        self.vehicle_type = VEHICLE_TYPES[vehicle_type]
        self.max_vehicles = max_vehicles
        self.time_limit_seconds = time_limit_seconds
        self.stops_per_parcel = stops_per_parcel
        self.fuel_price = fuel_price
        self.elec_price = elec_price

    def solve(
        self,
        assignment: "OperatorAssignment",
        city: "City",
        scenario: "Scenario | None" = None,
    ) -> VRPResult:
        try:
            from ortools.constraint_solver import pywrapcp, routing_enums_pb2
        except ImportError:
            raise ImportError(
                "ortools is required for VRPRouter:\n  pip install parcelsim[vrp]"
            )

        demand_factor = 1.0
        if scenario is not None:
            demand_factor = scenario.demand_factor * scenario.peak_factor

        scenario_name = scenario.name if scenario else "baseline"
        operator_results = []

        zones = assignment.zone_assignments
        depots_gdf = assignment.registry.all_depots(crs="EPSG:4326").to_crs(city.crs)

        for op in assignment.registry.operators:
            op_id = op.operator_id
            delivery_col = f"{op_id}_delivery"
            depot_col = f"{op_id}_depot"

            if delivery_col not in zones.columns:
                continue

            depot_results = []
            grouped = zones.groupby(depot_col)

            for depot_id, zone_group in grouped:
                depot_row = depots_gdf[depots_gdf["depot_id"] == depot_id]
                if depot_row.empty:
                    continue

                depot_x = depot_row.geometry.iloc[0].x
                depot_y = depot_row.geometry.iloc[0].y

                # Delivery stops: zones with demand > 0
                zone_group = zone_group[zone_group[delivery_col] > 0]
                if zone_group.empty:
                    depot_results.append(VRPDepotResult(
                        depot_id=str(depot_id),
                        operator_id=op_id,
                        n_stops=0,
                        n_vehicles=0,
                        vkt_km=0.0,
                    ))
                    continue

                # Node 0 = depot, nodes 1..N = zone centroids
                cx = zone_group["centroid_x"].values
                cy = zone_group["centroid_y"].values
                demands_raw = zone_group[delivery_col].values * demand_factor

                # Zone-level: demand = parcels in zone (rounded to int)
                zone_demands = np.maximum(1, np.round(demands_raw)).astype(int)

                n_nodes = len(zone_group) + 1  # +1 for depot
                xs = np.concatenate([[depot_x], cx])
                ys = np.concatenate([[depot_y], cy])

                dist_matrix = _euclidean_matrix_km(xs, ys)
                int_matrix = (dist_matrix * 1000).astype(int)  # metres (integer)

                # OR-Tools needs enough vehicles to serve all stops
                min_vehicles = math.ceil(zone_demands.sum() / self.vehicle_type.capacity_parcels)
                n_vehicles = min(self.max_vehicles, max(min_vehicles, 1))

                depot_result = self._solve_depot(
                    depot_id=str(depot_id),
                    operator_id=op_id,
                    n_vehicles=n_vehicles,
                    int_matrix=int_matrix,
                    zone_demands=zone_demands,
                    zone_ids=[str(z) for z in zone_group["zone_id"].values],
                    pywrapcp=pywrapcp,
                    routing_enums_pb2=routing_enums_pb2,
                )
                depot_results.append(depot_result)

            operator_results.append(VRPOperatorResult(
                operator_id=op_id,
                depot_results=depot_results,
            ))

        return VRPResult(operator_results=operator_results, scenario_name=scenario_name)

    def _solve_depot(
        self,
        depot_id: str,
        operator_id: str,
        n_vehicles: int,
        int_matrix: np.ndarray,
        zone_demands: np.ndarray,
        zone_ids: list[str],
        pywrapcp,
        routing_enums_pb2,
    ) -> VRPDepotResult:
        n_nodes = int_matrix.shape[0]
        vehicle_cap = self.vehicle_type.capacity_parcels

        demands_full = [0] + zone_demands.tolist()  # depot demand = 0

        manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_cb(from_idx, to_idx):
            return int(int_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])

        transit_idx = routing.RegisterTransitCallback(distance_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        def demand_cb(idx):
            return demands_full[manager.IndexToNode(idx)]

        demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
        routing.AddDimensionWithVehicleCapacity(
            demand_idx, 0,
            [vehicle_cap] * n_vehicles,
            True, "Capacity"
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_params.time_limit.seconds = self.time_limit_seconds

        solution = routing.SolveWithParameters(search_params)

        if solution is None:
            # Fallback: 2·r·m heuristic (same as CA access component)
            r_km = int_matrix[0, 1:].mean() / 1000 if n_nodes > 1 else 0.0
            m = n_vehicles
            vkt_km = 2 * r_km * m
            return VRPDepotResult(
                depot_id=depot_id,
                operator_id=operator_id,
                n_stops=len(zone_ids),
                n_vehicles=m,
                vkt_km=vkt_km,
                cost_eur=self._compute_cost(vkt_km, m),
            )

        routes: list[list[str]] = []
        total_dist_m = 0
        active_vehicles = 0

        for v in range(n_vehicles):
            idx = routing.Start(v)
            route_nodes = []
            route_dist = 0
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node > 0:
                    route_nodes.append(zone_ids[node - 1])
                next_idx = solution.Value(routing.NextVar(idx))
                route_dist += int_matrix[manager.IndexToNode(idx)][manager.IndexToNode(next_idx)]
                idx = next_idx
            if route_nodes:
                routes.append(route_nodes)
                total_dist_m += route_dist
                active_vehicles += 1

        vkt_km = total_dist_m / 1000.0

        return VRPDepotResult(
            depot_id=depot_id,
            operator_id=operator_id,
            n_stops=len(zone_ids),
            n_vehicles=active_vehicles,
            vkt_km=vkt_km,
            routes=routes,
            cost_eur=self._compute_cost(vkt_km, active_vehicles),
        )

    def _compute_cost(self, vkt_km: float, n_vehicles: int) -> float:
        """Total daily cost: salary + vehicle fixed + fuel/electricity."""
        vt = self.vehicle_type
        salary = n_vehicles * DRIVER_DAILY_SALARY_EUR
        vehicle_fixed = n_vehicles * vt.daily_fixed_cost_eur
        if vt.is_electric:
            energy_cost = vkt_km * vt.elec_wh_per_km / 1000.0 * self.elec_price
        else:
            energy_cost = vkt_km * vt.fuel_l_per_100km / 100.0 * self.fuel_price
        return salary + vehicle_fixed + energy_cost


def _euclidean_matrix_km(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix in km between all pairs of (x, y) points."""
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    return np.sqrt(dx ** 2 + dy ** 2) / 1000.0
