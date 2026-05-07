from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scenario:
    """
    Composable scenario modifier applied before routing.

    Scenarios do not require re-running the demand step unless demand_factor != 1.
    They are applied by the router at solve() time.
    """

    name: str
    demand_factor: float = 1.0
    peak_factor: float = 1.0
    fleet_overrides: dict = field(default_factory=dict)
    cost_overrides: dict = field(default_factory=dict)
    cargo_bike_policy: "CargoBikePolicy | None" = None

    def __add__(self, other: "Scenario") -> "Scenario":
        """Merge two scenarios. Later scenario's non-default values win."""
        return Scenario(
            name=f"{self.name}+{other.name}",
            demand_factor=other.demand_factor if other.demand_factor != 1.0 else self.demand_factor,
            peak_factor=other.peak_factor if other.peak_factor != 1.0 else self.peak_factor,
            fleet_overrides={**self.fleet_overrides, **other.fleet_overrides},
            cost_overrides={**self.cost_overrides, **other.cost_overrides},
            cargo_bike_policy=other.cargo_bike_policy or self.cargo_bike_policy,
        )


@dataclass
class CargoBikePolicy:
    """Parameters for cargo bike substitution analysis (Yang et al. 2024)."""
    eligible_fraction: float = 0.17
    max_parcels_per_stop: int = 7
    min_bike_lane_fraction: float = 0.10
