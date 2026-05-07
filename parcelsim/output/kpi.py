from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from parcelsim.emissions.factors import compute_ghg_kg, mtce_from_kg

if TYPE_CHECKING:
    from parcelsim.routing.ca.model import CARoutingResult
    from parcelsim.routing.vrp.result import VRPResult
    from parcelsim.operators.assignment import OperatorAssignment


@dataclass
class KPIReport:
    """Key performance indicators computed from a routing result."""

    scenario_name: str
    routing_mode: str

    # Demand
    total_parcels_delivered: float
    total_parcels_picked_up: float
    parcels_by_operator: dict[str, float]

    # Routing
    vkt_total_km: float
    vkt_by_operator: dict[str, float]
    n_trucks_total: int
    n_trucks_by_operator: dict[str, int]

    # Emissions
    ghg_kg_co2eq: float
    ghg_mtce: float
    ghg_g_per_parcel: float
    emission_factor_key: str

    # Cost (VRP mode only)
    cost_total: float | None = None
    cost_per_parcel: float | None = None

    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_ca(
        cls,
        routing_result: "CARoutingResult",
        assignment: "OperatorAssignment",
        emission_factor: str = "epa_light_truck",
    ) -> "KPIReport":
        demand = assignment.demand
        total_delivery = demand.effective_delivery
        total_pickup = demand.effective_pickup

        parcels_by_op = {}
        vkt_by_op = {}
        trucks_by_op = {}
        for op in assignment.registry.operators:
            col = f"{op.operator_id}_delivery"
            if col in assignment.zone_assignments.columns:
                parcels_by_op[op.operator_id] = float(
                    assignment.zone_assignments[col].sum()
                )

        for r in routing_result.operator_results:
            vkt_by_op[r.operator_id] = r.vkt_total_km
            trucks_by_op[r.operator_id] = r.n_trucks_total

        vkt_total = routing_result.vkt_total_km
        ghg_kg = compute_ghg_kg(vkt_total, emission_factor)
        ghg_mtce = mtce_from_kg(ghg_kg)
        ghg_per_parcel = (ghg_kg / total_delivery * 1000) if total_delivery > 0 else 0.0

        return cls(
            scenario_name=routing_result.scenario_name,
            routing_mode="ca",
            total_parcels_delivered=total_delivery,
            total_parcels_picked_up=total_pickup,
            parcels_by_operator=parcels_by_op,
            vkt_total_km=vkt_total,
            vkt_by_operator=vkt_by_op,
            n_trucks_total=routing_result.n_trucks_total,
            n_trucks_by_operator=trucks_by_op,
            ghg_kg_co2eq=ghg_kg,
            ghg_mtce=ghg_mtce,
            ghg_g_per_parcel=ghg_per_parcel,
            emission_factor_key=emission_factor,
        )

    @classmethod
    def from_vrp(
        cls,
        routing_result: "VRPResult",
        assignment: "OperatorAssignment",
        emission_factor: str = "fr_icv_small",
    ) -> "KPIReport":
        demand = assignment.demand
        total_delivery = demand.effective_delivery
        total_pickup = demand.effective_pickup

        parcels_by_op = {}
        for op in assignment.registry.operators:
            col = f"{op.operator_id}_delivery"
            if col in assignment.zone_assignments.columns:
                parcels_by_op[op.operator_id] = float(
                    assignment.zone_assignments[col].sum()
                )

        vkt_by_op = {r.operator_id: r.vkt_total_km for r in routing_result.operator_results}
        trucks_by_op = {r.operator_id: r.n_vehicles_total for r in routing_result.operator_results}
        cost_by_op = {r.operator_id: r.cost_total_eur for r in routing_result.operator_results}

        vkt_total = routing_result.vkt_total_km
        ghg_kg = compute_ghg_kg(vkt_total, emission_factor)
        ghg_mtce = mtce_from_kg(ghg_kg)
        ghg_per_parcel = (ghg_kg / total_delivery * 1000) if total_delivery > 0 else 0.0
        cost_total = routing_result.cost_total_eur
        cost_per_parcel = cost_total / total_delivery if total_delivery > 0 else None

        return cls(
            scenario_name=routing_result.scenario_name,
            routing_mode="vrp",
            total_parcels_delivered=total_delivery,
            total_parcels_picked_up=total_pickup,
            parcels_by_operator=parcels_by_op,
            vkt_total_km=vkt_total,
            vkt_by_operator=vkt_by_op,
            n_trucks_total=routing_result.n_vehicles_total,
            n_trucks_by_operator=trucks_by_op,
            ghg_kg_co2eq=ghg_kg,
            ghg_mtce=ghg_mtce,
            ghg_g_per_parcel=ghg_per_parcel,
            emission_factor_key=emission_factor,
            cost_total=cost_total,
            cost_per_parcel=cost_per_parcel,
            metadata={"cost_by_operator": cost_by_op},
        )

    def summary(self) -> str:
        sep = "─" * 50
        lines = [
            sep,
            f"KPIReport  [{self.routing_mode.upper()}]  scenario={self.scenario_name}",
            sep,
            f"  Parcels delivered:  {self.total_parcels_delivered:>12,.0f} /day",
            f"  Parcels picked up:  {self.total_parcels_picked_up:>12,.0f} /day",
            f"  Total VKT:          {self.vkt_total_km:>12,.1f} km/day",
            f"  Active trucks:      {self.n_trucks_total:>12,}",
            f"  GHG emissions:      {self.ghg_kg_co2eq:>12,.1f} kg CO₂eq/day",
            f"  GHG (MTCE):         {self.ghg_mtce:>12.2f} MTCE/day",
            f"  GHG per parcel:     {self.ghg_g_per_parcel:>12.1f} g CO₂eq",
            "  By operator:",
        ]
        for op_id, vkt in self.vkt_by_operator.items():
            trucks = self.n_trucks_by_operator.get(op_id, 0)
            parcels = self.parcels_by_operator.get(op_id, 0)
            lines.append(
                f"    {op_id:10s}  parcels={parcels:>8,.0f}  "
                f"VKT={vkt:>9,.1f} km  trucks={trucks:>4}"
            )
        lines.append(sep)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "routing_mode": self.routing_mode,
            "total_parcels_delivered": self.total_parcels_delivered,
            "total_parcels_picked_up": self.total_parcels_picked_up,
            "vkt_total_km": self.vkt_total_km,
            "n_trucks_total": self.n_trucks_total,
            "ghg_kg_co2eq": self.ghg_kg_co2eq,
            "ghg_mtce": self.ghg_mtce,
            "ghg_g_per_parcel": self.ghg_g_per_parcel,
            "vkt_by_operator": self.vkt_by_operator,
            "n_trucks_by_operator": self.n_trucks_by_operator,
            "parcels_by_operator": self.parcels_by_operator,
        }

    def to_parquet(self, output_dir: Path | str) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame([self.to_dict()])
        summary_df.to_parquet(output_dir / f"{self.scenario_name}_kpi.parquet", index=False)

        by_op = pd.DataFrame({
            "operator_id": list(self.vkt_by_operator.keys()),
            "vkt_km": list(self.vkt_by_operator.values()),
            "n_trucks": [self.n_trucks_by_operator.get(k, 0) for k in self.vkt_by_operator],
            "parcels": [self.parcels_by_operator.get(k, 0) for k in self.vkt_by_operator],
        })
        by_op.to_parquet(output_dir / f"{self.scenario_name}_by_operator.parquet", index=False)
