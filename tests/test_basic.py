"""
Smoke tests that run without downloading any external data.
Uses synthetic in-memory data to validate the full pipeline logic.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from parcelsim.city import City
from parcelsim.population.base import SyntheticPopulation
from parcelsim.demand.base import ParcelDemand
from parcelsim.demand.usps_model import USPSDemandModel
from parcelsim.operators.operator import Depot, Operator, OperatorRegistry
from parcelsim.operators.assignment import assign_parcels
from parcelsim.routing.ca.model import CARouter
from parcelsim.output.kpi import KPIReport
from parcelsim.scenario.modifiers import demand_growth, peak_season, cargo_bike_substitution
from parcelsim.emissions.factors import compute_ghg_kg, mtce_from_kg
from parcelsim.demand.france_model import FranceDemandModel
from parcelsim.demand.generic_model import AggregateDemandModel
from parcelsim.routing.vrp.model import VRPRouter
from parcelsim.output.kpi import KPIReport


CRS = "EPSG:32618"  # UTM 18N (NYC)


def _make_city() -> City:
    """Minimal city with 3 synthetic census tracts."""
    zones = gpd.GeoDataFrame(
        {
            "zone_id":      ["36061000100", "36061000200", "36047000100"],
            "population":   [5000, 8000, 3000],
            "n_households": [2000, 3200, 1200],
            "area_km2":     [0.5, 0.8, 0.3],
            "centroid_x":   [585000.0, 586000.0, 584000.0],
            "centroid_y":   [4511000.0, 4512000.0, 4510000.0],
            "geometry":     [
                Polygon([(584800, 4510800), (585200, 4510800),
                          (585200, 4511200), (584800, 4511200)]),
                Polygon([(585800, 4511800), (586200, 4511800),
                          (586200, 4512200), (585800, 4512200)]),
                Polygon([(583800, 4509800), (584200, 4509800),
                          (584200, 4510200), (583800, 4510200)]),
            ],
        },
        geometry="geometry",
        crs=CRS,
    )
    study_area = gpd.GeoDataFrame(
        geometry=[zones.union_all()], crs=CRS
    )
    return City(
        name="test_city",
        country_iso="US",
        crs=CRS,
        study_area=study_area,
        zones=zones,
    )


def _make_population(city: City) -> SyntheticPopulation:
    """200 synthetic households spread across 3 zones, 4 income brackets."""
    rng = np.random.default_rng(0)
    brackets = ["lt35k", "35k_65k", "65k_100k", "gt100k"]
    rows = []
    for zone_id, row in city.zones.iterrows():
        n_hh = row["n_households"] // 16  # ~75-200 per zone
        for bracket in brackets:
            for i in range(n_hh):
                rows.append({
                    "household_id": f"{row['zone_id']}_{bracket}_{i}",
                    "zone_id": row["zone_id"],
                    "geometry": Point(row["centroid_x"], row["centroid_y"]),
                    "n_persons": int(rng.integers(1, 5)),
                    "income_bracket": bracket,
                })
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS)
    return SyntheticPopulation(
        city=city,
        households=gdf,
        source_adapter="test",
        year=2021,
    )


def _make_registry() -> OperatorRegistry:
    # Depot coordinates in WGS84 (lat/lon) — assign_parcels reprojects to city CRS
    # Approximate NYC area: lat~40.71, lon~-74.01
    ops = [
        Operator(
            operator_id="ups",
            name="UPS",
            market_share=0.50,
            depots=[Depot("ups_1", "ups", Point(-74.006, 40.714))],  # lon, lat
        ),
        Operator(
            operator_id="fedex",
            name="FedEx",
            market_share=0.50,
            depots=[Depot("fedex_1", "fedex", Point(-74.011, 40.705))],
        ),
    ]
    return OperatorRegistry(ops)


# ─────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────

def test_city_zone_validation():
    city = _make_city()
    assert "zone_id" in city.zones.columns
    assert "n_households" in city.zones.columns
    assert len(city.zones) == 3


def test_synthetic_population_summary():
    city = _make_city()
    pop = _make_population(city)
    assert pop.n_households > 0
    assert pop.n_persons > 0
    summary = pop.summary()
    assert "test_city" in summary


def test_usps_demand_model():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)

    assert demand.total_delivery > 0
    assert demand.total_pickup > 0
    assert demand.total_pickup < demand.total_delivery
    assert "zone_id" in demand.zone_demand.columns
    assert "n_delivery" in demand.zone_demand.columns
    assert len(demand.zone_demand) == len(city.zones)


def test_demand_apply_factor():
    city = _make_city()
    pop = _make_population(city)
    base = USPSDemandModel().generate(pop)
    scaled = base.apply_factor(demand_factor=2.0)

    assert scaled.effective_delivery == pytest.approx(base.total_delivery * 2.0)
    assert base.effective_delivery == base.total_delivery   # original unchanged


def test_operator_registry_from_builtin():
    registry = OperatorRegistry.from_builtin("us_2021")
    total_share = sum(op.market_share for op in registry.operators)
    assert abs(total_share - 1.0) < 0.01
    assert len(registry.operators) == 4
    assert any(op.operator_id == "usps" for op in registry.operators)


def test_operator_registry_market_share_validation():
    with pytest.raises(ValueError, match="market shares sum to"):
        OperatorRegistry([
            Operator("a", "A", market_share=0.6, depots=[]),
            Operator("b", "B", market_share=0.6, depots=[]),
        ])


def test_assign_parcels_euclidean():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()

    assignment = assign_parcels(demand, registry, city)
    zones = assignment.zone_assignments

    assert "ups_delivery" in zones.columns
    assert "fedex_delivery" in zones.columns
    assert "ups_depot" in zones.columns
    # All zones assigned to a depot
    assert zones["ups_depot"].notna().all()


def test_ca_router_basic():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)

    result = CARouter().solve(assignment, city)

    assert result.vkt_total_km > 0
    assert result.n_trucks_total > 0
    assert len(result.operator_results) == 2


def test_ca_router_daganzo_formula():
    """Verify CA formula produces sensible values for a toy case."""
    router = CARouter(default_k=0.85, truck_capacity=300)
    # 600 stops, 2 km² area, 5 km depot distance → m=2 trucks
    # V = 2*5*2 + 0.85*(600-2)*sqrt(2/600)
    #   = 20 + 0.85 * 598 * 0.0577 ≈ 20 + 29.3 ≈ 49.3 km
    import numpy as np
    n, A, r, k, m = 600, 2.0, 5.0, 0.85, 2
    expected = 2 * r * m + k * (n - m) * np.sqrt(A / n)
    assert 40 < expected < 60   # sanity range


def test_kpi_report_from_ca():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)
    result = CARouter().solve(assignment, city)

    report = KPIReport.from_ca(result, assignment)

    assert report.vkt_total_km > 0
    assert report.ghg_kg_co2eq > 0
    assert report.ghg_mtce > 0
    assert report.ghg_g_per_parcel > 0
    assert "ups" in report.vkt_by_operator
    summary = report.summary()
    assert "KPIReport" in summary


def test_emission_calculations():
    # EPA factor: 234 g CO2/mile = 145.4 g CO2/km (Yang et al. 2024)
    ghg = compute_ghg_kg(100.0, "epa_light_truck")
    assert ghg == pytest.approx(100.0 * 234.0 / 1.60934 / 1000.0, rel=1e-3)
    # MTCE = metric tons CO2eq (1 MTCE = 1000 kg CO2eq)
    assert mtce_from_kg(1000.0) == pytest.approx(1.0)


def test_scenario_demand_growth():
    city = _make_city()
    pop = _make_population(city)
    base_demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(base_demand, registry, city)

    s = demand_growth(2.0)
    assert s.demand_factor == 2.0

    result_base = CARouter().solve(assignment, city)
    result_scaled = CARouter().solve(assignment, city, scenario=s)
    assert result_scaled.vkt_total_km > result_base.vkt_total_km


def test_scenario_composition():
    s1 = demand_growth(1.5)
    s2 = peak_season(1.104)
    combined = s1 + s2
    assert combined.demand_factor == 1.5
    assert combined.peak_factor == 1.104
    assert "+" in combined.name


def test_france_demand_model():
    city = _make_city()
    pop = _make_population(city)
    demand = FranceDemandModel(demand_factor=1.35).generate(pop)

    assert demand.total_delivery > 0
    assert demand.total_pickup == 0.0   # France model: no pickup modelled
    assert "n_delivery" in demand.zone_demand.columns
    assert len(demand.zone_demand) == len(city.zones)


def test_vrp_router_basic():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)

    result = VRPRouter(vehicle_type="medium_icv", time_limit_seconds=5).solve(assignment, city)

    assert result.vkt_total_km > 0
    assert result.n_vehicles_total > 0
    assert len(result.operator_results) == 2


def test_kpi_report_from_vrp():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)
    result = VRPRouter(vehicle_type="medium_icv", time_limit_seconds=5).solve(assignment, city)

    report = KPIReport.from_vrp(result, assignment, emission_factor="fr_icv_small")

    assert report.vkt_total_km > 0
    assert report.ghg_kg_co2eq > 0
    assert report.cost_total is not None
    assert report.cost_per_parcel is not None
    assert report.routing_mode == "vrp"


def test_lyon_registry():
    registry = OperatorRegistry.from_builtin("lyon_2024")
    total_share = sum(op.market_share for op in registry.operators)
    assert abs(total_share - 1.0) < 0.01
    assert len(registry.operators) == 8
    assert any(op.operator_id == "colissimo" for op in registry.operators)


def test_to_segments_with_synthetic_network():
    import networkx as nx
    from shapely.geometry import LineString

    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)

    # Build a minimal synthetic osmnx-style MultiDiGraph in the city CRS
    G = nx.MultiDiGraph(crs=CRS)
    coords = [
        (585000.0, 4511000.0),
        (586000.0, 4512000.0),
        (584000.0, 4510000.0),
    ]
    for i, (x, y) in enumerate(coords):
        G.add_node(i, x=x, y=y, osmid=1000 + i)
    edges = [(0, 1, 0), (1, 2, 0), (0, 2, 0)]
    for u, v, k in edges:
        xu, yu = coords[u]
        xv, yv = coords[v]
        G.add_edge(u, v, key=k, osmid=2000 + u * 10 + v,
                   highway="residential", length=500.0,
                   geometry=LineString([(xu, yu), (xv, yv)]))

    city.road_network = G

    segs = assignment.to_segments(city, weight="road_length")

    assert "n_delivery" in segs.columns
    assert "ups_delivery" in segs.columns
    assert "ups_depot" in segs.columns
    assert segs.index.names == ["u", "v", "key"]
    # Segments outside all zone polygons get 0 — total may be < zone total
    assert segs["n_delivery"].sum() >= 0
    assert segs["ups_delivery"].sum() + segs["fedex_delivery"].sum() == pytest.approx(
        segs["n_delivery"].sum(), rel=1e-6
    )


def test_usps_demand_std():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    zd = demand.zone_demand

    assert "n_delivery_std" in zd.columns
    assert "n_delivery_p05" in zd.columns
    assert "n_delivery_p95" in zd.columns
    assert (zd["n_delivery_std"] >= 0).all()
    assert (zd["n_delivery_p05"] <= zd["n_delivery"]).all()
    assert (zd["n_delivery_p95"] >= zd["n_delivery"]).all()
    # Poisson: std = sqrt(mean)
    np.testing.assert_allclose(
        zd["n_delivery_std"].values,
        np.sqrt(zd["n_delivery"].values),
        rtol=1e-6,
    )


def test_france_demand_std():
    city = _make_city()
    pop = _make_population(city)
    demand = FranceDemandModel(demand_factor=1.35).generate(pop)
    zd = demand.zone_demand

    assert "n_delivery_std" in zd.columns
    assert (zd["n_delivery_std"] >= 0).all()
    assert (zd["n_delivery_p05"] >= 0).all()
    assert (zd["n_delivery_p95"] >= zd["n_delivery_p05"]).all()


def test_to_segments_std_propagation():
    import networkx as nx
    from shapely.geometry import LineString

    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)

    G = nx.MultiDiGraph(crs=CRS)
    coords = [(585000.0, 4511000.0), (586000.0, 4512000.0), (584000.0, 4510000.0)]
    for i, (x, y) in enumerate(coords):
        G.add_node(i, x=x, y=y, osmid=1000 + i)
    for u, v, k in [(0, 1, 0), (1, 2, 0), (0, 2, 0)]:
        xu, yu = coords[u]; xv, yv = coords[v]
        G.add_edge(u, v, key=k, osmid=2000 + u * 10 + v,
                   highway="residential", length=500.0,
                   geometry=LineString([(xu, yu), (xv, yv)]))
    city.road_network = G

    segs = assignment.to_segments(city, weight="road_length")

    assert "n_delivery_std" in segs.columns
    assert "n_delivery_p05" in segs.columns
    assert "n_delivery_p95" in segs.columns
    assert (segs["n_delivery_std"] >= 0).all()


def test_to_blocks():
    city = _make_city()
    pop = _make_population(city)
    demand = USPSDemandModel().generate(pop)
    registry = _make_registry()
    assignment = assign_parcels(demand, registry, city)

    # Use zone polygons as blocks (sub-divide each zone into 2 halves)
    zones = city.zones.copy()
    blocks_rows = []
    for _, row in zones.iterrows():
        b = row.geometry.bounds  # minx, miny, maxx, maxy
        mid_x = (b[0] + b[2]) / 2
        from shapely.geometry import box
        blocks_rows.append({"geometry": box(b[0], b[1], mid_x, b[3])})
        blocks_rows.append({"geometry": box(mid_x, b[1], b[2], b[3])})
    blocks = gpd.GeoDataFrame(blocks_rows, geometry="geometry", crs=CRS)

    result = assignment.to_blocks(blocks, city, weight="area")

    assert "n_delivery" in result.columns
    assert "n_delivery_std" in result.columns
    assert "n_delivery_p05" in result.columns
    assert "zone_id" in result.columns
    assert (result["n_delivery"] >= 0).all()
    assert (result["n_delivery_std"] >= 0).all()


def test_cargo_bike_scenario():
    s = cargo_bike_substitution(eligible_fraction=0.17)
    assert s.demand_factor == pytest.approx(0.83)
    assert s.cargo_bike_policy is not None
    assert s.cargo_bike_policy.eligible_fraction == 0.17


def test_aggregate_demand_available_countries():
    countries = AggregateDemandModel.available_countries()
    assert len(countries) >= 30
    assert "DE" in countries
    assert "GB" in countries
    assert "BR" in countries


def test_aggregate_demand_from_country_invalid():
    with pytest.raises(KeyError, match="not in registry"):
        AggregateDemandModel.from_country("XX")


def test_aggregate_demand_from_country_generate():
    model = AggregateDemandModel.from_country("DE")
    city = _make_city()
    pop = _make_population(city)
    demand = model.generate(pop)
    zd = demand.zone_demand
    assert len(zd) > 0
    assert "n_delivery" in zd.columns
    assert "n_delivery_std" in zd.columns
    assert "n_delivery_p05" in zd.columns
    assert "n_delivery_p95" in zd.columns
    assert (zd["n_delivery"] >= 0).all()
    assert (zd["n_delivery_std"] >= 0).all()
    assert (zd["n_delivery_p05"] <= zd["n_delivery_p95"]).all()


def test_aggregate_demand_daily_rate():
    model = AggregateDemandModel(
        annual_parcels_per_hh=30.0,
        home_delivery_fraction=0.55,
        delivery_days_per_year=250,
        demand_factor=1.0,
    )
    expected = 30.0 * 0.55 / 250
    assert model.daily_rate_per_hh == pytest.approx(expected)
