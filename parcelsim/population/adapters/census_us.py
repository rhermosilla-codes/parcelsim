from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

from parcelsim.population.base import SyntheticPopulation

if TYPE_CHECKING:
    from parcelsim.city import City

# ACS 5-year table B19001: Household Income in the Past 12 Months
# Variables bracketed into 4 groups matching Yang et al. (2024)
_INCOME_BRACKET_MAP: dict[str, list[str]] = {
    "lt35k":    [f"B19001_{str(i).zfill(3)}E" for i in range(2, 8)],   # <$35k
    "35k_65k":  [f"B19001_{str(i).zfill(3)}E" for i in range(8, 13)],  # $35k–$65k (up to $74,999)
    "65k_100k": ["B19001_013E"],                                         # $75k–$99,999
    "gt100k":   [f"B19001_{str(i).zfill(3)}E" for i in range(14, 18)],  # ≥$100k
}

_ALL_INCOME_VARS = [v for vs in _INCOME_BRACKET_MAP.values() for v in vs]

# Average household size by income bracket (US national averages, ACS 2020)
_AVG_HH_SIZE: dict[str, float] = {
    "lt35k":    2.1,
    "35k_65k":  2.6,
    "65k_100k": 2.9,
    "gt100k":   3.1,
}


class USCensusAdapter:
    """
    Builds a SyntheticPopulation from US Census ACS 5-year data.

    Requires: pip install parcelsim[us]  (censusdis)

    Households are allocated at census tract level. If land_use_source="uniform",
    the household point is placed at the tract centroid (adequate for CA routing).
    """

    def __init__(
        self,
        state: str,
        county_fips: list[str],
        acs_year: int = 2020,
        land_use_source: str = "uniform",
        census_api_key: str | None = None,
        cache_dir: Path | str = "./parcelsim_cache",
    ) -> None:
        self.state = state
        self.county_fips = county_fips
        self.acs_year = acs_year
        self.land_use_source = land_use_source
        self.census_api_key = census_api_key
        self.cache_dir = Path(cache_dir)

    def build(self, city: "City") -> SyntheticPopulation:
        try:
            import censusdis.data as ced
            from censusdis.datasets import ACS5
        except ImportError:
            raise ImportError(
                "censusdis is required for the US Census adapter:\n"
                "  pip install parcelsim[us]"
            )

        cache_path = self._cache_path()
        if cache_path.exists():
            tracts = gpd.read_parquet(cache_path)
        else:
            tracts = self._download(ced, ACS5)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tracts.to_parquet(cache_path)

        households = self._build_households(tracts, city.crs)
        city.zones = self._build_zones(tracts, city.crs)

        return SyntheticPopulation(
            city=city,
            households=households,
            source_adapter="census_us",
            year=self.acs_year,
            metadata={
                "state": self.state,
                "counties": self.county_fips,
                "n_tracts": len(tracts),
                "acs_year": self.acs_year,
            },
        )

    def _download(self, ced, ACS5) -> gpd.GeoDataFrame:
        download_vars = ["NAME", "B01003_001E", "B11001_001E"] + _ALL_INCOME_VARS

        frames = []
        for county in self.county_fips:
            df = ced.download(
                dataset=ACS5,
                vintage=self.acs_year,
                download_variables=download_vars,
                state=_state_fips(self.state),
                county=county,
                tract="*",
                with_geometry=True,
            )
            frames.append(df)

        tracts = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True))
        tracts = tracts.rename(columns={
            "B01003_001E": "population",
            "B11001_001E": "n_households",
        })

        for bracket, vars_ in _INCOME_BRACKET_MAP.items():
            cols = [v for v in vars_ if v in tracts.columns]
            tracts[f"hh_{bracket}"] = tracts[cols].clip(lower=0).sum(axis=1)

        tracts["zone_id"] = tracts["STATE"] + tracts["COUNTY"] + tracts["TRACT"]
        return tracts[tracts["n_households"] > 0].reset_index(drop=True)

    def _build_zones(self, tracts: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
        zones = tracts[["zone_id", "geometry", "population", "n_households"]].copy()
        zones = zones.to_crs(crs)
        zones["area_km2"] = zones.geometry.area / 1e6
        centroid = zones.geometry.centroid
        zones["centroid_x"] = centroid.x
        zones["centroid_y"] = centroid.y
        return zones.reset_index(drop=True)

    def _build_households(self, tracts: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
        rows = []
        rng = np.random.default_rng(42)

        tracts_proj = tracts.to_crs(crs)

        for _, tract in tracts_proj.iterrows():
            zone_id = tract["zone_id"]
            centroid = tract.geometry.centroid

            for bracket in _INCOME_BRACKET_MAP:
                n = int(tract.get(f"hh_{bracket}", 0))
                if n == 0:
                    continue
                avg_size = _AVG_HH_SIZE[bracket]
                n_persons = rng.integers(
                    max(1, round(avg_size) - 1),
                    round(avg_size) + 2,
                    size=n,
                )
                for i in range(n):
                    rows.append({
                        "household_id": f"{zone_id}_{bracket}_{i}",
                        "zone_id": zone_id,
                        "geometry": centroid,
                        "n_persons": int(n_persons[i]),
                        "income_bracket": bracket,
                    })

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
        return gdf.reset_index(drop=True)

    def _cache_path(self) -> Path:
        counties = "_".join(sorted(self.county_fips))
        return self.cache_dir / f"census_us_{self.state}_{counties}_{self.acs_year}.parquet"


def _state_fips(state: str) -> str:
    """Accept state abbrev or FIPS code."""
    _abbrev_to_fips = {
        "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
        "CO": "08", "CT": "09", "DE": "10", "FL": "12", "GA": "13",
        "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
        "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
        "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29",
        "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34",
        "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
        "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45",
        "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50",
        "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56",
        "DC": "11",
    }
    if state.isdigit() and len(state) == 2:
        return state
    return _abbrev_to_fips.get(state.upper(), state)
