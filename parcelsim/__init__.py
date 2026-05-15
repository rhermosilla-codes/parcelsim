"""
parcelsim — city-agnostic synthetic last-mile parcel delivery demand generation.

Based on:
  Yang, Landes & Chow (2024) — NYC Continuous Approximation model
  Hörl, Briand & Puchinger (2025) — Lyon replicable VRP pipeline

Quick start (US / NYC):
    import parcelsim as ps
    from parcelsim.population.adapters.census_us import USCensusAdapter
    from parcelsim.demand.usps_model import USPSDemandModel
    from parcelsim.operators.operator import OperatorRegistry
    from parcelsim.operators.assignment import assign_parcels
    from parcelsim.routing.ca.model import CARouter
    from parcelsim.output.kpi import KPIReport

    city       = ps.City.from_osmnx("New York City, New York, USA", crs="EPSG:32618")
    population = USCensusAdapter(state="NY", county_fips=["061","047","081","005","085"]).build(city)
    demand     = USPSDemandModel().generate(population)
    registry   = OperatorRegistry.from_builtin("us_2021")
    assignment = assign_parcels(demand, registry, city)
    result     = CARouter().solve(assignment, city)
    report     = KPIReport.from_ca(result, assignment)
    print(report.summary())
"""

from parcelsim.city import City

__version__ = "0.1.1"
__all__ = ["City", "__version__"]
