from __future__ import annotations

from dataclasses import dataclass

# EPA emission standard for light delivery trucks: 234 g CO₂/mile (EPA 2021)
# Yang et al. (2024) use this factor and report MTCE as metric tons CO₂ (not carbon-only).
# Converted to g/km: 234 / 1.60934 = 145.4 g CO₂/km
EPA_LIGHT_TRUCK_G_CO2_PER_MILE = 234.0
EPA_LIGHT_TRUCK_G_CO2_PER_KM = EPA_LIGHT_TRUCK_G_CO2_PER_MILE / 1.60934  # 145.4 g/km

# BEV emission factor for France (90 gCO2eq/kWh, low due to nuclear)
# Source: RTE France / Hörl et al. (2025)
FR_BEV_G_CO2EQ_PER_KWH = 90.0

# US grid average emission factor (EIA 2021, ~385 gCO2/kWh)
US_BEV_G_CO2EQ_PER_KWH = 385.0


@dataclass
class EmissionFactor:
    g_co2eq_per_km: float
    source: str
    vehicle_type: str = "diesel_light_truck"


BUILTIN_FACTORS: dict[str, EmissionFactor] = {
    "epa_light_truck": EmissionFactor(
        g_co2eq_per_km=EPA_LIGHT_TRUCK_G_CO2_PER_KM,
        source="EPA (2021)",
        vehicle_type="diesel_light_truck",
    ),
    "fr_bev_small": EmissionFactor(
        g_co2eq_per_km=14.4,   # 160 Wh/km × 90 gCO2eq/kWh
        source="Hörl et al. (2025) Table 4",
        vehicle_type="bev_small",
    ),
    "fr_bev_medium": EmissionFactor(
        g_co2eq_per_km=18.0,
        source="Hörl et al. (2025) Table 4",
        vehicle_type="bev_medium",
    ),
    "fr_icv_small": EmissionFactor(
        g_co2eq_per_km=130.0,
        source="Hörl et al. (2025) Table 4",
        vehicle_type="icv_small",
    ),
}


def compute_ghg_kg(vkt_km: float, factor_key: str = "epa_light_truck") -> float:
    """Convert vehicle-km to kg CO2eq using a named emission factor."""
    factor = BUILTIN_FACTORS.get(factor_key)
    if factor is None:
        raise KeyError(f"Unknown emission factor: {factor_key}. "
                       f"Available: {list(BUILTIN_FACTORS)}")
    return vkt_km * factor.g_co2eq_per_km / 1000.0


def mtce_from_kg(kg_co2eq: float) -> float:
    """Convert kg CO2eq to metric tons CO2 equivalent (MTCE). 1 MTCE = 1 metric ton CO2eq."""
    return kg_co2eq / 1000.0
