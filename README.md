# `parcelsim`

City-agnostic Python library for generating synthetic last-mile parcel delivery demand and routing KPIs. Supports any city worldwide through pluggable population adapters, two demand models, and both Continuous Approximation (CA) and Vehicle Routing Problem (VRP) routing engines.

Built on top of two peer-reviewed pipelines:

- **Yang, Landes & Chow (2024)** — Continuous Approximation model calibrated to NYC ([paper](papers/1-s2.0-S2046043023000692-main.pdf))
- **Horl, Briand & Puchinger (2025)** — Replicable VRP pipeline for Lyon Metropole ([paper](papers/1-s2.0-S1361920925003037-main.pdf))

---

## Authors

- **Rodrigo Hermosilla P**, PhD in AI & ML | Machine Learning Research Engineer, 
Intelligent Logistics and Sustainability (ILS) · Center for Transportation & Logistics (CTL) · MIT
</br>
- **Juan C. Pina-Pardo**, PhD | Assistant Professor, Pontificia Universidad Católica de Valparaíso | Research Affiliate, Massachusetts Institute of Technology <br/>
- **Philipp Zinnenlauf**, PhD Student Systems Engineering (CEE) · Center for Transportation & Logistics (CTL) · MIT

---

## Installation

```bash
pip install parcelsim
```

**With optional extras:**

```bash
pip install "parcelsim[us]"          # + US Census population adapter
pip install "parcelsim[vrp]"         # + OR-Tools CVRP solver
pip install "parcelsim[worldpop]"    # + WorldPop raster adapter
pip install "parcelsim[us,vrp]"      # full install
```

| Extra      | Installs                  | Use case                     |
|------------|---------------------------|------------------------------|
| `us`       | `censusdis`               | US Census population adapter |
| `vrp`      | `ortools`                 | OR-Tools CVRP solver         |
| `worldpop` | `rasterstats`, `rasterio` | WorldPop raster adapter      |

---

## Quick Start

### Case 1 — NYC with US Census data (CA routing)

Replicates Yang et al. (2024). Downloads ACS 2020 census tracts for the five NYC boroughs and runs the Continuous Approximation (CA) formula.

```python
from parcelsim.city import City
from parcelsim.population.adapters.census_us import USCensusAdapter
from parcelsim.demand.usps_model import USPSDemandModel
from parcelsim.operators.operator import OperatorRegistry
from parcelsim.operators.assignment import assign_parcels
from parcelsim.routing.ca.model import CARouter
from parcelsim.output.kpi import KPIReport

city = City.from_osmnx("New York City, New York, USA", crs="EPSG:32618")

population = USCensusAdapter(
    state="NY",
    county_fips=["061", "047", "081", "005", "085"],  # Manhattan + 4 boroughs
    acs_year=2020,
).build(city)

demand     = USPSDemandModel().generate(population)
registry   = OperatorRegistry.from_builtin("us_2021")   # USPS, UPS, FedEx, Amazon
assignment = assign_parcels(demand, registry, city)
result     = CARouter().solve(assignment, city)
report     = KPIReport.from_ca(result, assignment)

print(report.summary())
# KPIReport  [CA]  scenario=baseline
#   Parcels delivered:   3,456,789 /day
#   Total VKT:           8,234.1 km/day
#   GHG emissions:       1,197.5 kg CO2eq/day
```

### Case 2 — Lyon with WorldPop raster + VRP routing

Replicates Horl et al. (2025). Downloads WorldPop 1 km population raster for France
(no Census API required) and solves a capacitated VRP with OR-Tools.

```python
from parcelsim.city import City
from parcelsim.population.adapters.worldpop import WorldPopAdapter
from parcelsim.demand.france_model import FranceDemandModel
from parcelsim.operators.operator import OperatorRegistry
from parcelsim.operators.assignment import assign_parcels
from parcelsim.routing.vrp.model import VRPRouter
from parcelsim.output.kpi import KPIReport

city = City.from_osmnx("Lyon, France", crs="EPSG:2154")

population = WorldPopAdapter(country_iso2="FR", year=2020).build(city)

demand     = FranceDemandModel(demand_factor=1.35).generate(population)  # 2024 baseline
registry   = OperatorRegistry.from_builtin("lyon_2024")  # 8 French carriers
assignment = assign_parcels(demand, registry, city)
result     = VRPRouter(vehicle_type="medium_icv", time_limit_seconds=30).solve(assignment, city)
report     = KPIReport.from_vrp(result, assignment, emission_factor="fr_icv_small")

print(report.summary())
print(f"Cost per parcel: {report.cost_per_parcel:.2f} EUR")
```

### Case 3 — Fully synthetic data (no downloads, CI/testing)

This approach requires no internet access. Build a city from scratch with synthetic zone polygons —
the approach used in all unit tests.

```python
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from parcelsim.city import City
from parcelsim.population.base import SyntheticPopulation
from parcelsim.demand.usps_model import USPSDemandModel
from parcelsim.operators.operator import Operator, Depot, OperatorRegistry
from parcelsim.operators.assignment import assign_parcels
from parcelsim.routing.ca.model import CARouter
from parcelsim.output.kpi import KPIReport

CRS = "EPSG:32618"

zones = gpd.GeoDataFrame({
    "zone_id":      ["zone_A", "zone_B"],
    "population":   [50000, 80000],
    "n_households": [20000, 32000],
    "area_km2":     [3.0, 5.0],
    "centroid_x":   [585000.0, 586500.0],
    "centroid_y":   [4511000.0, 4512000.0],
    "geometry": [
        Polygon([(584800,4510800),(585200,4510800),(585200,4511200),(584800,4511200)]),
        Polygon([(586300,4511800),(586700,4511800),(586700,4512200),(586300,4512200)]),
    ],
}, geometry="geometry", crs=CRS)

city = City(
    name="test_city", country_iso="US", crs=CRS,
    study_area=gpd.GeoDataFrame(geometry=[zones.union_all()], crs=CRS),
    zones=zones,
)

rows = [{"household_id": f"hh_{i}", "zone_id": "zone_A",
         "geometry": Point(585000, 4511000),
         "n_persons": 3, "income_bracket": "35k_65k"}
        for i in range(500)]
population = SyntheticPopulation(
    city=city,
    households=gpd.GeoDataFrame(rows, geometry="geometry", crs=CRS),
    source_adapter="synthetic",
    year=2021,
)

demand     = USPSDemandModel().generate(population)
registry   = OperatorRegistry([
    Operator("ups",   "UPS",   market_share=0.5,
             depots=[Depot("ups_1",   "ups",   Point(-74.006, 40.714))]),
    Operator("fedex", "FedEx", market_share=0.5,
             depots=[Depot("fedex_1", "fedex", Point(-74.011, 40.705))]),
])
assignment = assign_parcels(demand, registry, city)
result     = CARouter().solve(assignment, city)
report     = KPIReport.from_ca(result, assignment)
print(report.summary())
```

### Case 4 — Scenario analysis and comparison

Compose scenarios with `+` and compare them in a single loop.
`assignment` and `city` are built as shown in Cases 1–3.

```python
from typing import cast
from parcelsim.city import City
from parcelsim.operators.assignment import OperatorAssignment
from parcelsim.routing.ca.model import CARouter
from parcelsim.output.kpi import KPIReport
from parcelsim.scenario.modifiers import demand_growth, peak_season, cargo_bike_substitution

city       = cast(City, cast(object, ...))             # defined in Cases 1–3
assignment = cast(OperatorAssignment, cast(object, ...))  # defined in Cases 1–3

scenarios = {
    "baseline":          None,
    "2030_bau":          demand_growth(2.0),
    "2030_peak":         demand_growth(2.0) + peak_season(1.104),
    "cargo_bikes_17pct": cargo_bike_substitution(eligible_fraction=0.17),
}

results = {}
for name, scenario in scenarios.items():
    result = CARouter().solve(assignment, city, scenario=scenario)
    results[name] = KPIReport.from_ca(result, assignment)
    print(f"{name:20s}  VKT={results[name].vkt_total_km:>8,.0f} km  "
          f"GHG={results[name].ghg_kg_co2eq:>8,.0f} kg")
```

### Case 5 — Custom operator registry from YAML

```yaml
# my_city_2024.yaml
operators:
  - id: carrier_a
    name: "City Carrier A"
    market_share: 0.60
    depots:
      - id: depot_north
        lon: -73.950
        lat: 40.780
  - id: carrier_b
    name: "City Carrier B"
    market_share: 0.40
    depots:
      - id: depot_south
        lon: -73.980
        lat: 40.700
```

```python
from parcelsim.operators.operator import OperatorRegistry

registry = OperatorRegistry.from_yaml("my_city_2024.yaml")
```

---

## Pipeline Overview

```
City  --► PopulationAdapter --► SyntheticPopulation
                                       |
                               DemandModel.generate()
                                       |
                            ParcelDemand  (mean + std per zone)
                                       |
                           assign_parcels(demand, registry, city)
                                       |
                              OperatorAssignment
                         +------------|------------+
                    to_segments()  to_blocks()   CARouter / VRPRouter
                         |              |               |
                    segment GDF    block GDF        KPIReport
```

Every step produces a typed, inspectable object. The router applies scenarios at `solve()` time — the demand and assignment steps do not need to be repeated.

---

## Demand Models

### USPS model — `USPSDemandModel` (Yang et al. 2024)

Income-stratified household parcel generation. Requires households with an `income_bracket`
column (`lt35k`, `35k_65k`, `65k_100k`, `gt100k`).

```
daily deliveries = sum_i (N_i * U_i * F) / (omega_p * d)
```

| Parameter            | Value                           | Source               |
|----------------------|---------------------------------|----------------------|
| `parcel_rate`        | 0.19 parcel/hh/day              | USPS 2021            |
| Income multipliers   | 0.7x (lt35k) – 1.6x (gt100k)    | Yang et al. Table 1  |
| Pickup fraction      | 0.66                            | USPS 2021            |

### Generic model — `AggregateDemandModel`

Country-agnostic aggregate model. Parameters are loaded from the built-in registry by ISO code.
Demand rates are calibrated for 30 countries — but operator presets (carriers, depots) exist only
for France and the US. For any other country you must add an operator YAML before running the
full pipeline (see [Extending the Library](#extending-the-library)).

```python
from parcelsim.demand.generic_model import AggregateDemandModel

model  = AggregateDemandModel.from_country("KR", demand_factor=1.0)
demand = model.generate(population)
# demand.zone_demand includes n_delivery, n_delivery_std, n_delivery_p05, n_delivery_p95
```

See `parcelsim/demand/builtin/country_params.yaml` for all rate parameters and sources.

---

### France model — `FranceDemandModel` (Horl et al. 2025)

Aggregate household rate from the Gardrat (2019) national survey. No income stratification
is needed — compatible with WorldPop.

```
mu_hh = annual_purchases * demand_factor * home_delivery_fraction / delivery_days
      = 14 * 1.35 * 0.47 / 260 = 0.034 parcels/hh/day  (2024 baseline)
```

| `demand_factor` | Year | Source                  |
|-----------------|------|-------------------------|
| 1.00            | 2019 | Gardrat (2019)          |
| 1.35            | 2024 | Horl et al. Table 5     |
| 2.00            | 2030 | Horl et al. Table 5     |

---

## Sub-Zone Disaggregation

Both demand models produce probabilistic output — every zone gets a point estimate **and** a distribution:

```python
demand.zone_demand[["n_delivery", "n_delivery_std", "n_delivery_p05", "n_delivery_p95"]]
```

Uncertainty model: daily zone demand follows a Poisson distribution with mean = point estimate. `std = sqrt(mean)`. The 5th and 95th percentiles use a Normal approximation (valid for zones with >30 households).

### Street segment level — `to_segments()`

Requires `city.road_network` (any `osmnx` MultiDiGraph, including a pre-computed `G_primal`). Output is indexed by `(u, v, key)` — the same index as `osmnx` edges, enabling a direct join.

```python
city.road_network = G_primal   # Philipp's graph, or any osmnx graph
segments = assignment.to_segments(city, weight="road_length")
# columns: osmid, geometry, zone_id, seg_weight,
#          n_delivery, n_delivery_std, n_delivery_p05, n_delivery_p95,
#          {op}_delivery, {op}_depot
```

Std propagation: `segment_std = sqrt(seg_weight) * zone_std` (Poisson sub-sampling property).

### City block level — `to_blocks()`

Accepts any GeoDataFrame of polygons — OSM blocks, H3 hexagons, custom grids.

```python
import h3pandas  # optional — any polygon GDF works
blocks = gpd.read_file("boston_blocks.geojson")
result = assignment.to_blocks(blocks, city, weight="area")
# columns: geometry, zone_id, block_weight,
#          n_delivery, n_delivery_std, n_delivery_p05, n_delivery_p95,
#          {op}_delivery, {op}_depot
```

---

## Routing Engines

### Continuous Approximation — `CARouter`

CA formula, calibrated per NYC borough. Fast — no solver required.

```
V = 2 * r * m + k * (n - m) * sqrt(A / n)
```

| Symbol | Meaning                                                           |
|--------|-------------------------------------------------------------------|
| `V`    | Vehicle-km-traveled per depot service area                        |
| `r`    | Depot to service area centroid distance (km)                      |
| `m`    | Number of trucks = ceil(n / truck_capacity)                       |
| `n`    | Total delivery stops                                              |
| `A`    | Service area (km2)                                                |
| `k`    | Network coefficient: MN=0.708, BX=0.894, BK/QN=0.856, SI=0.993    |

```python
from typing import cast
from parcelsim.city import City
from parcelsim.operators.assignment import OperatorAssignment
from parcelsim.routing.ca.model import CARouter

city       = cast(City, cast(object, ...))
assignment = cast(OperatorAssignment, cast(object, ...))

result = CARouter(default_k=0.85, truck_capacity=300).solve(assignment, city)
```

### VRP — `VRPRouter` (requires `ortools`)

Google OR-Tools CVRP, one instance per depot-operator pair. Zone centroids serve as stops;
Euclidean distances in the projected CRS. Includes a cost model (salary + vehicle fixed +
fuel/electricity).

```python
from typing import cast
from parcelsim.city import City
from parcelsim.operators.assignment import OperatorAssignment
from parcelsim.routing.vrp.model import VRPRouter

city       = cast(City, cast(object, ...))
assignment = cast(OperatorAssignment, cast(object, ...))

result = VRPRouter(
    vehicle_type="medium_icv",    # 50 parcels/vehicle, 260 EUR/month
    max_vehicles=40,
    time_limit_seconds=30,
).solve(assignment, city)
```

Available vehicle types (Horl et al. Table 4):

| Key          | Capacity | Monthly cost | Emissions     |
|--------------|----------|--------------|---------------|
| `small_icv`  | 33       | 210 EUR      | 130 g CO2/km  |
| `medium_icv` | 50       | 260 EUR      | 160 g CO2/km  |
| `large_icv`  | 100      | 370 EUR      | 215 g CO2/km  |
| `small_bev`  | 33       | 260 EUR      | 14.4 g CO2/km |
| `medium_bev` | 50       | 400 EUR      | 18.0 g CO2/km |
| `large_bev`  | 100      | 800 EUR      | 27.0 g CO2/km |

---

## Built-in Operator Presets

| Preset       | Operators                                                            | Coverage                            |
|--------------|----------------------------------------------------------------------|-------------------------------------|
| `us_2021`    | USPS, UPS, FedEx, Amazon                                             | NYC baseline (Yang et al. 2024)     |
| `lyon_2024`  | Colissimo, Chronopost, UPS, DPD, DHL, GLS, Colis Prive, FedEx        | Lyon Metropole (Horl et al. 2025)   |

Add your own preset in `parcelsim/operators/builtin/<preset>.yaml`. Depot coordinates are WGS84 (lon, lat).

---

## Emission Factors

| Key                | Factor          | Source                       |
|--------------------|-----------------|------------------------------|
| `epa_light_truck`  | 145.4 g CO2/km  | EPA (2021), 234 g CO2/mile   |
| `fr_icv_small`     | 130.0 g CO2/km  | Horl et al. (2025) Table 4   |
| `fr_bev_small`     | 14.4 g CO2eq/km | Horl et al. (2025) Table 4   |
| `fr_bev_medium`    | 18.0 g CO2eq/km | Horl et al. (2025) Table 4   |

BEV factors use the French grid (90 g CO2eq/kWh, nuclear-heavy). Add country-specific
factors in `parcelsim/emissions/factors.py`.

---

## Notebooks

| Notebook                            | Description                                                              |
|-------------------------------------|--------------------------------------------------------------------------|
| `notebooks/demo_synthetic.ipynb`    | Full NYC pipeline with synthetic data, no downloads                      |
| `notebooks/validation_nyc.ipynb`    | Validation against Yang et al. (2024) Tables 1 and 4                     |
| `notebooks/demo_lyon.ipynb`         | Full Lyon pipeline: WorldPop population, France demand, 8-operator VRP   |

---

## Tests

```bash
pytest                          # all 27 tests
pytest tests/test_basic.py -v   # verbose
pytest -k test_vrp_router       # single test
```

All tests use in-memory synthetic data and require no internet access.

---

## Extending the Library

### Add a new city

1. Implement a `PopulationAdapter` (see `census_us.py` as reference). It must return a
   `SyntheticPopulation` with an `income_bracket` column, or use `FranceDemandModel`
   which does not require income data.
2. Add a demand model or reuse an existing one.
3. Add an operator YAML in `operators/builtin/<preset>.yaml`.
4. Test offline using the `_make_city()` / `_make_population()` pattern from
   `tests/test_basic.py`.

### Add a demand model for a new country

#### Built-in demand rates — `AggregateDemandModel`

`AggregateDemandModel.from_country(iso)` gives you a calibrated daily household parcel rate for
30 countries. That covers the **demand step only** — to run the full pipeline you also need an
operator preset (`operators/builtin/`) for local carriers and depots. Currently only France and
the US have operator presets.

```python
from parcelsim.demand.generic_model import AggregateDemandModel

model  = AggregateDemandModel.from_country("DE", demand_factor=1.1)
demand = model.generate(population)
```

Countries with demand rates (ISO codes):

| Region        | Demand rate | Operator preset |
|---------------|-------------|-----------------|
| AT, BE, CH, CZ, DE, DK, ES, FI, FR, GB, HU, IT, NL, NO, PL, PT, RO, SE | ✓ | FR only |
| CA, MX, US    | ✓           | US only         |
| AU, JP, KR, NZ, SG | ✓     | —               |
| AR, BR, CL, CO | ✓          | —               |

```python
# Inspect all rate parameters
AggregateDemandModel.available_countries()
```

Rate parameters live in `parcelsim/demand/builtin/country_params.yaml` — extend without touching Python code.

#### Adding a country not yet in the registry

**Option A — Aggregate rate (quickest)**

Add a block to `country_params.yaml`:

```yaml
IN:
  annual_parcels_per_hh: 4.0
  home_delivery_fraction: 0.88
  delivery_days_per_year: 300
  source: "India Post Annual Report (2022)"
```

Then call `AggregateDemandModel.from_country("IN")`. No Python changes needed.

**Option B — Income-stratified (Census or survey data available)**

Create a new model following `USPSDemandModel` as a template. You need:
- Parcel generation rates per income bracket (weekly parcels/household)
- National carrier market shares

**Checklist**

1. Find an annual e-commerce parcel survey (postal operator report, Eurostat, national statistics office)
2. Identify: parcels/household/year, home delivery fraction, delivery days/year
3. Add to `country_params.yaml` (Option A) or implement a stratified model (Option B)
4. Add an operator YAML in `operators/builtin/<country>_<year>.yaml` with local carriers and market shares
5. Add country-specific emission factors in `emissions/factors.py`
6. Test offline with `_make_city()` — no downloads needed

### Add a new emission factor

```python
from parcelsim.emissions.factors import BUILTIN_FACTORS, EmissionFactor

BUILTIN_FACTORS["eu_heavy_diesel"] = EmissionFactor(
    g_co2eq_per_km=270.0,
    source="EEA (2023)",
    vehicle_type="diesel_heavy_truck",
)
```

---

## Project Structure

```
parcelsim/
├── city.py                          # City dataclass — canonical spatial unit
├── population/
│   ├── base.py                      # SyntheticPopulation dataclass
│   └── adapters/
│       ├── census_us.py             # US Census ACS adapter
│       └── worldpop.py              # WorldPop 1 km raster adapter
├── demand/
│   ├── base.py                      # ParcelDemand dataclass
│   ├── usps_model.py                # Yang et al. (2024) income-stratified model
│   └── france_model.py              # Horl et al. (2025) aggregate model
├── operators/
│   ├── operator.py                  # Operator, Depot, OperatorRegistry
│   ├── assignment.py                # assign_parcels() — nearest-depot allocation
│   └── builtin/                     # YAML presets: us_2021, lyon_2024
├── routing/
│   ├── ca/model.py                  # CARouter — Daganzo formula
│   └── vrp/                         # VRPRouter — OR-Tools CVRP
├── scenario/
│   ├── base.py                      # Scenario dataclass (composable with +)
│   └── modifiers.py                 # demand_growth, peak_season, cargo_bike_substitution
├── emissions/factors.py             # Emission factor registry
└── output/kpi.py                    # KPIReport — from_ca() and from_vrp()
```

---

## References

Yang, X., Landes, B., & Chow, J. Y. J. (2024). Synthetic last-mile parcel delivery demand
generation for New York City using Continuous Approximation. *Research in Transportation
Business & Management*, 53, 101090.

Horl, S., Briand, A., & Puchinger, J. (2025). A replicable pipeline for last-mile parcel
delivery simulation in Lyon Metropole. *Transportation Research Part A*, 191, 104275.
