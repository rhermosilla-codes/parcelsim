from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

from parcelsim.population.base import SyntheticPopulation

if TYPE_CHECKING:
    from parcelsim.city import City

# WorldPop unconstrained individual counts, 1 km resolution
_WORLDPOP_URL = (
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/"
    "{year}/{iso3_upper}/{iso3_lower}_ppp_{year}_1km_Aggregated_UNadj.tif"
)

# Average household size by country ISO2 (fallback = 2.5)
_AVG_HH_SIZE: dict[str, float] = {
    "FR": 2.18,
    "US": 2.53,
    "GB": 2.36,
    "DE": 2.04,
    "ES": 2.50,
    "IT": 2.35,
    "BR": 3.10,
    "CN": 2.90,
    "IN": 4.40,
}


class WorldPopAdapter:
    """
    Builds a SyntheticPopulation from WorldPop 1-km raster population data.

    Works for any city worldwide — no income differentiation (use with
    FranceDemandModel or other aggregate-rate models).

    Requires: pip install parcelsim[worldpop]  (rasterio, rasterstats)

    Parameters
    ----------
    country_iso2 : str
        Two-letter ISO country code (e.g. "FR", "US").
    year : int
        Population year. WorldPop provides 2000-2020.
    avg_hh_size : float | None
        Average household size. If None, uses country default or 2.5.
    cache_dir : Path | str
        Directory for caching the downloaded GeoTIFF.
    """

    def __init__(
        self,
        country_iso2: str,
        year: int = 2020,
        avg_hh_size: float | None = None,
        cache_dir: Path | str = "./parcelsim_cache",
    ) -> None:
        self.country_iso2 = country_iso2.upper()
        self.year = year
        self.avg_hh_size = avg_hh_size or _AVG_HH_SIZE.get(self.country_iso2, 2.5)
        self.cache_dir = Path(cache_dir)

    def build(self, city: "City") -> SyntheticPopulation:
        try:
            import rasterio
            from rasterstats import zonal_stats
        except ImportError:
            raise ImportError(
                "rasterio and rasterstats are required for WorldPopAdapter:\n"
                "  pip install parcelsim[worldpop]"
            )

        raster_path = self._get_raster(city)
        zones = city.zones.copy()
        if zones.empty:
            raise ValueError("city.zones must be populated before calling WorldPopAdapter.build()")

        zones_wgs84 = zones.to_crs("EPSG:4326")
        stats = zonal_stats(
            zones_wgs84,
            str(raster_path),
            stats=["sum"],
            nodata=-9999,
            all_touched=True,
        )
        pop_counts = [max(0.0, s["sum"] or 0.0) for s in stats]

        zones = zones.copy()
        zones["population"] = [int(round(p)) for p in pop_counts]
        zones["n_households"] = [max(1, int(round(p / self.avg_hh_size))) for p in pop_counts]
        if "area_km2" not in zones.columns:
            zones["area_km2"] = zones.geometry.area / 1e6
        if "centroid_x" not in zones.columns:
            centroid = zones.geometry.centroid
            zones["centroid_x"] = centroid.x
            zones["centroid_y"] = centroid.y

        city.zones = zones

        households = self._build_households(zones)
        return SyntheticPopulation(
            city=city,
            households=households,
            source_adapter="worldpop",
            year=self.year,
            metadata={
                "country_iso2": self.country_iso2,
                "worldpop_year": self.year,
                "avg_hh_size": self.avg_hh_size,
                "n_zones": len(zones),
            },
        )

    def _build_households(self, zones: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        rows = []
        for _, zone in zones.iterrows():
            zid = zone["zone_id"]
            n = int(zone["n_households"])
            if n == 0:
                continue
            rows.append({
                "household_id": f"{zid}_median_0",
                "zone_id": zid,
                "geometry": zone.geometry.centroid,
                "n_persons": round(self.avg_hh_size),
                "income_bracket": "median",
                "n_households": n,
            })
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=zones.crs)

    def _get_raster(self, city: "City") -> Path:
        iso3 = _iso2_to_iso3(self.country_iso2)
        fname = f"{iso3.lower()}_ppp_{self.year}_1km_Aggregated_UNadj.tif"
        cache_path = self.cache_dir / "worldpop" / fname

        if cache_path.exists():
            return cache_path

        url = _WORLDPOP_URL.format(
            year=self.year,
            iso3_upper=iso3.upper(),
            iso3_lower=iso3.lower(),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading WorldPop raster for {self.country_iso2} {self.year}...")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, cache_path)
        print(f"  Saved to {cache_path}")
        return cache_path


_ISO2_TO_ISO3 = {
    "FR": "FRA", "US": "USA", "GB": "GBR", "DE": "DEU", "ES": "ESP",
    "IT": "ITA", "BR": "BRA", "CN": "CHN", "IN": "IND", "MX": "MEX",
    "JP": "JPN", "KR": "KOR", "AU": "AUS", "CA": "CAN", "NL": "NLD",
    "BE": "BEL", "CH": "CHE", "AT": "AUT", "PL": "POL", "SE": "SWE",
    "NO": "NOR", "DK": "DNK", "PT": "PRT", "GR": "GRC", "CZ": "CZE",
    "HU": "HUN", "RO": "ROU", "ZA": "ZAF", "NG": "NGA", "EG": "EGY",
    "AR": "ARG", "CL": "CHL", "CO": "COL", "PE": "PER",
}


def _iso2_to_iso3(iso2: str) -> str:
    code = _ISO2_TO_ISO3.get(iso2.upper())
    if code is None:
        raise ValueError(f"Unknown ISO2 country code: {iso2}. Add it to _ISO2_TO_ISO3.")
    return code
