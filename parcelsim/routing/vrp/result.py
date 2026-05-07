from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VRPDepotResult:
    """Routing result for a single depot."""

    depot_id: str
    operator_id: str
    n_stops: int
    n_vehicles: int
    vkt_km: float
    cost_eur: float = 0.0
    routes: list[list[str]] = field(default_factory=list)


@dataclass
class VRPOperatorResult:
    """Aggregated VRP result across all depots of one operator."""

    operator_id: str
    depot_results: list[VRPDepotResult]

    @property
    def vkt_total_km(self) -> float:
        return sum(d.vkt_km for d in self.depot_results)

    @property
    def n_vehicles_total(self) -> int:
        return sum(d.n_vehicles for d in self.depot_results)

    @property
    def cost_total_eur(self) -> float:
        return sum(d.cost_eur for d in self.depot_results)

    @property
    def n_stops_total(self) -> int:
        return sum(d.n_stops for d in self.depot_results)


@dataclass
class VRPResult:
    """System-wide VRP routing result, analogous to CARoutingResult."""

    operator_results: list[VRPOperatorResult]
    scenario_name: str = "baseline"

    @property
    def vkt_total_km(self) -> float:
        return sum(r.vkt_total_km for r in self.operator_results)

    @property
    def n_vehicles_total(self) -> int:
        return sum(r.n_vehicles_total for r in self.operator_results)

    @property
    def cost_total_eur(self) -> float:
        return sum(r.cost_total_eur for r in self.operator_results)

    def summary(self) -> str:
        lines = [
            f"VRPResult  scenario={self.scenario_name}",
            f"  Total VKT:    {self.vkt_total_km:>12,.1f} km/day",
            f"  Total vehicles: {self.n_vehicles_total:>10,}",
            f"  Total cost:   {self.cost_total_eur:>12,.0f} EUR/day",
            "  By operator:",
        ]
        for r in self.operator_results:
            lines.append(
                f"    {r.operator_id:12s}  VKT={r.vkt_total_km:>9,.1f} km"
                f"  vehicles={r.n_vehicles_total:>4}"
                f"  cost={r.cost_total_eur:>8,.0f} EUR"
            )
        return "\n".join(lines)
