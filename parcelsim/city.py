from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import networkx as nx


@dataclass
class City:
    """Root geographic context passed through every pipeline step."""

    name: str
    country_iso: str
    crs: str
    study_area: gpd.GeoDataFrame
    zones: gpd.GeoDataFrame
    road_network: nx.MultiDiGraph | None = None
    delivery_days_per_year: int = 260
    cache_dir: Path = field(default_factory=lambda: Path("./parcelsim_cache"))

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._validate_zones()

    def _validate_zones(self) -> None:
        if self.zones.empty:
            return  # adapter will populate zones later
        required = {"zone_id", "geometry", "population", "n_households"}
        missing = required - set(self.zones.columns)
        if missing:
            raise ValueError(f"zones GeoDataFrame missing required columns: {missing}")

    @classmethod
    def from_osmnx(
        cls,
        query: str,
        crs: str,
        country_iso: str = "",
        cache_dir: Path | str = "./parcelsim_cache",
        network_type: str = "drive",
        load_network: bool = True,
    ) -> "City":
        """
        Build a City from an OpenStreetMap place query.

        zones will be empty until a population adapter populates them.
        Call city.zones = adapter.build(city).zones to populate.
        """
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("osmnx is required: pip install osmnx")

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        ox.settings.cache_folder = str(cache_dir / "osmnx")
        ox.settings.use_cache = True

        study_area = ox.geocode_to_gdf(query).to_crs(crs)

        road_network = None
        if load_network:
            road_network = ox.graph_from_place(query, network_type=network_type)
            road_network = ox.project_graph(road_network, to_crs=crs)
            road_network = ox.add_edge_speeds(road_network)
            road_network = ox.add_edge_travel_times(road_network)

        # Empty zones — populated later by a PopulationAdapter
        empty_zones = gpd.GeoDataFrame(
            columns=["zone_id", "geometry", "population", "n_households"],
            geometry="geometry",
            crs=crs,
        )

        return cls(
            name=query,
            country_iso=country_iso or _guess_iso(query),
            crs=crs,
            study_area=study_area,
            zones=empty_zones,
            road_network=road_network,
            cache_dir=cache_dir,
        )

    @classmethod
    def from_zones(
        cls,
        name: str,
        zones: gpd.GeoDataFrame,
        crs: str,
        country_iso: str = "",
        cache_dir: Path | str = "./parcelsim_cache",
    ) -> "City":
        """Build a City directly from a pre-processed zones GeoDataFrame."""
        if zones.crs is None or str(zones.crs) != crs:
            zones = zones.to_crs(crs)
        return cls(
            name=name,
            country_iso=country_iso,
            crs=crs,
            study_area=gpd.GeoDataFrame(
                geometry=[zones.union_all()], crs=crs
            ),
            zones=zones,
            cache_dir=cache_dir,
        )


def _guess_iso(query: str) -> str:
    q = query.upper()
    if any(x in q for x in ["USA", "UNITED STATES", "NEW YORK", "LOS ANGELES", "CHICAGO"]):
        return "US"
    if any(x in q for x in ["FRANCE", "PARIS", "LYON", "MARSEILLE"]):
        return "FR"
    if any(x in q for x in ["BRAZIL", "BRASIL", "SÃO PAULO", "RIO"]):
        return "BR"
    return ""
