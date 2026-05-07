from __future__ import annotations

from parcelsim.scenario.base import CargoBikePolicy, Scenario


def demand_growth(factor: float, name: str | None = None) -> Scenario:
    """Scale total parcel demand by a multiplicative factor."""
    return Scenario(name=name or f"demand_{factor}x", demand_factor=factor)


def peak_season(factor: float = 1.104, name: str = "peak_season") -> Scenario:
    """Apply peak season volume factor (1.104 = US average, Yang et al. Table 2)."""
    return Scenario(name=name, peak_factor=factor)


def carbon_tax(eur_per_tco2: float) -> Scenario:
    """Add carbon cost to vehicle operating costs (Hörl et al. 2025)."""
    return Scenario(
        name=f"carbon_tax_{int(eur_per_tco2)}EUR",
        cost_overrides={"carbon_tax_eur_per_tco2": eur_per_tco2},
    )


def icv_purchase_tax(factor: float) -> Scenario:
    """Multiply ICV monthly vehicle cost by (1 + factor)."""
    return Scenario(
        name=f"icv_tax_{int(factor * 100)}pct",
        fleet_overrides={"icv_purchase_tax_factor": factor},
    )


def full_electric_mandate() -> Scenario:
    """Remove all ICV vehicle types from available fleet."""
    return Scenario(
        name="full_electric",
        fleet_overrides={"exclude_propulsion": ["ICV"]},
    )


def cargo_bike_substitution(
    eligible_fraction: float = 0.17,
    max_parcels_per_stop: int = 7,
    min_bike_lane_fraction: float = 0.10,
) -> Scenario:
    """
    Cargo bike substitution scenario — Yang et al. (2024) Section 4.4.

    eligible_fraction: share of delivery stops replaced by cargo bikes
    max_parcels_per_stop: max volume per stop for bike eligibility
    min_bike_lane_fraction: minimum bike lane % in service area for eligibility
    """
    return Scenario(
        name=f"cargo_bikes_{int(eligible_fraction * 100)}pct",
        demand_factor=1.0 - eligible_fraction,  # reduce truck demand by substituted fraction
        cargo_bike_policy=CargoBikePolicy(
            eligible_fraction=eligible_fraction,
            max_parcels_per_stop=max_parcels_per_stop,
            min_bike_lane_fraction=min_bike_lane_fraction,
        ),
    )
