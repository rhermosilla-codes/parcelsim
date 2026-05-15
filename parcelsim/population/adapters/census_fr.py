from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import shape

from parcelsim.population.base import SyntheticPopulation

if TYPE_CHECKING:
    from parcelsim.city import City

# Average household size in France (INSEE RP 2020)
_AVG_HH_SIZE_FR = 2.19

# Filosofi income deciles → income bracket mapping (EUR/year, disposable income)
# Brackets calibrated to match the spirit of Yang et al. (2024) US income brackets
_INCOME_THRESHOLDS = {
    "lt22k":    (0,     22_000),
    "22k_40k":  (22_000, 40_000),
    "40k_60k":  (40_000, 60_000),
    "gt60k":    (60_000, float("inf")),
}

_OPENDATASOFT_IRIS = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets"
    "/georef-france-iris/exports/json"
)
_INSEE_POP_URLS: dict[int, str] = {
    2020: "https://www.insee.fr/fr/statistiques/fichier/7704076/base-ic-evol-struct-pop-2020_csv.zip",
    2019: "https://www.insee.fr/fr/statistiques/fichier/6543200/base-ic-evol-struct-pop-2019_csv.zip",
}
_FILOSOFI_URLS: dict[int, str] = {
    2020: "https://www.insee.fr/fr/statistiques/fichier/7233950/BASE_TD_FILO_DEC_IRIS_2020_CSV.zip",
}


class FranceCensusAdapter:
    """
    Builds a SyntheticPopulation from French INSEE IRIS-level data.

    Data sources (no registration required):
    - IRIS geometries: OpenDataSoft / IGN (filtered by department)
    - Population & households: INSEE Recensement de la Population (RP) 2020
    - Income distribution: INSEE Filosofi 2020 (income deciles per IRIS)

    Household count is estimated from P20_PMEN (population in ordinary
    households) divided by the French average household size (2.19 in 2020).

    Usage::

        from parcelsim.population.adapters.census_fr import FranceCensusAdapter

        adapter = FranceCensusAdapter(departements=["69"], year=2020)
        population = adapter.build(city)
    """

    def __init__(
        self,
        departements: list[str],
        year: int = 2020,
        communes: list[str] | None = None,
        cache_dir: Path | str = "./parcelsim_cache",
    ) -> None:
        self.departements = [str(d).zfill(2) for d in departements]
        self.year = year
        self.communes = communes
        self.cache_dir = Path(cache_dir)

    def build(self, city: "City") -> SyntheticPopulation:
        cache_path = self._cache_path()
        if cache_path.exists():
            iris = gpd.read_parquet(cache_path)
        else:
            iris = self._download()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            iris.to_parquet(cache_path)

        if self.communes:
            iris = iris[iris["com_code"].isin(self.communes)].reset_index(drop=True)

        households = self._build_households(iris, city.crs)
        city.zones = self._build_zones(iris, city.crs)

        return SyntheticPopulation(
            city=city,
            households=households,
            source_adapter="census_fr",
            year=self.year,
            metadata={
                "departements": self.departements,
                "communes": self.communes,
                "n_iris": len(iris),
                "year": self.year,
            },
        )

    # ------------------------------------------------------------------ #
    # Download                                                             #
    # ------------------------------------------------------------------ #

    def _download(self) -> gpd.GeoDataFrame:
        print(f"Downloading IRIS geometries for departments {self.departements}...")
        geom = self._download_geometries()

        print(f"Downloading INSEE RP {self.year} demographics...")
        demo = self._download_demographics()

        print(f"Downloading Filosofi {self.year} income data...")
        filo = self._download_filosofi()

        # Merge on 9-digit IRIS code
        iris = geom.merge(demo, on="iris_code", how="left")
        iris = iris.merge(filo, on="iris_code", how="left")

        iris["n_households"] = (
            iris["pop_in_hh"].fillna(0) / _AVG_HH_SIZE_FR
        ).clip(lower=0).round().astype(int)
        iris["population"] = iris["population"].fillna(0).astype(int)

        return iris[iris["n_households"] > 0].reset_index(drop=True)

    def _download_geometries(self) -> gpd.GeoDataFrame:
        dep_filter = " OR ".join(f"dep_code='{d}'" for d in self.departements)
        params = {
            "where": dep_filter,
            "select": "iris_code,iris_name,iris_type,com_code,com_name,geo_shape",
            "limit": -1,
        }
        try:
            resp = requests.get(_OPENDATASOFT_IRIS, params=params, timeout=60)
            resp.raise_for_status()
            records = resp.json()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download IRIS geometries from OpenDataSoft: {exc}"
            ) from exc

        rows = []
        for rec in records:
            gs = rec.get("geo_shape")
            if not gs:
                continue
            geom = gs.get("geometry") or gs
            iris_code = rec.get("iris_code")
            if isinstance(iris_code, list):
                iris_code = iris_code[0]
            rows.append({
                "iris_code":  iris_code,
                "iris_name":  (rec.get("iris_name") or [""])[0] if isinstance(rec.get("iris_name"), list) else rec.get("iris_name", ""),
                "iris_type":  (rec.get("iris_type") or [""])[0] if isinstance(rec.get("iris_type"), list) else rec.get("iris_type", ""),
                "com_code":   (rec.get("com_code") or [""])[0] if isinstance(rec.get("com_code"), list) else rec.get("com_code", ""),
                "com_name":   (rec.get("com_name") or [""])[0] if isinstance(rec.get("com_name"), list) else rec.get("com_name", ""),
                "geometry":   shape(geom),
            })

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
        return gdf

    def _download_demographics(self) -> pd.DataFrame:
        url = _INSEE_POP_URLS.get(self.year)
        if url is None:
            raise ValueError(
                f"No INSEE RP URL configured for year {self.year}. "
                f"Supported years: {sorted(_INSEE_POP_URLS)}"
            )
        try:
            resp = requests.get(url, timeout=120, headers={"User-Agent": "parcelsim"})
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(f"Failed to download INSEE RP data: {exc}") from exc

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(f for f in z.namelist() if f.endswith(".CSV") and "meta" not in f)
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1",
                             usecols=["IRIS", "COM", "TYP_IRIS", "P20_POP", "P20_PMEN"],
                             dtype=str)

        # Filter to requested departments
        dep_set = set(self.departements)
        df = df[df["IRIS"].str[:2].isin(dep_set)].copy()

        df["P20_POP"]  = pd.to_numeric(df["P20_POP"],  errors="coerce").fillna(0)
        df["P20_PMEN"] = pd.to_numeric(df["P20_PMEN"], errors="coerce").fillna(0)

        df = df.rename(columns={
            "IRIS":     "iris_code",
            "COM":      "com_code_insee",
            "TYP_IRIS": "iris_type_insee",
            "P20_POP":  "population",
            "P20_PMEN": "pop_in_hh",
        })
        return df[["iris_code", "population", "pop_in_hh"]]

    def _download_filosofi(self) -> pd.DataFrame:
        url = _FILOSOFI_URLS.get(self.year)
        if url is None:
            return pd.DataFrame(columns=["iris_code", "median_income_eur",
                                         "d1_income", "d9_income"])
        try:
            resp = requests.get(url, timeout=60, headers={"User-Agent": "parcelsim"})
            resp.raise_for_status()
        except Exception:
            return pd.DataFrame(columns=["iris_code", "median_income_eur",
                                         "d1_income", "d9_income"])

        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = next(f for f in z.namelist() if f.endswith(".csv") and "meta" not in f.lower())
        with z.open(csv_name) as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1",
                             usecols=["IRIS", "DEC_MED20", "DEC_D120", "DEC_D920"],
                             dtype=str)

        dep_set = set(self.departements)
        df = df[df["IRIS"].str[:2].isin(dep_set)].copy()

        for col in ["DEC_MED20", "DEC_D120", "DEC_D920"]:
            df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")

        df = df.rename(columns={
            "IRIS":       "iris_code",
            "DEC_MED20":  "median_income_eur",
            "DEC_D120":   "d1_income",
            "DEC_D920":   "d9_income",
        })
        return df[["iris_code", "median_income_eur", "d1_income", "d9_income"]]

    # ------------------------------------------------------------------ #
    # Build households / zones                                            #
    # ------------------------------------------------------------------ #

    def _build_zones(self, iris: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
        zones = iris[["iris_code", "geometry", "population", "n_households"]].copy()
        zones = zones.rename(columns={"iris_code": "zone_id"})
        zones = zones.to_crs(crs)
        zones["area_km2"] = zones.geometry.area / 1e6
        centroid = zones.geometry.centroid
        zones["centroid_x"] = centroid.x
        zones["centroid_y"] = centroid.y
        return zones.reset_index(drop=True)

    def _build_households(self, iris: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
        iris_proj = iris.to_crs(crs)
        rows = []
        for _, zone in iris_proj.iterrows():
            zone_id  = zone["iris_code"]
            n_hh     = int(zone["n_households"])
            centroid = zone.geometry.centroid
            med_inc  = zone.get("median_income_eur", np.nan)

            bracket = _income_bracket(med_inc)
            rows.append({
                "household_id":   zone_id,
                "zone_id":        zone_id,
                "geometry":       centroid,
                "n_persons":      2,
                "income_bracket": bracket,
                "n_households":   n_hh,
            })

        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs).reset_index(drop=True)

    def _cache_path(self) -> Path:
        deps = "_".join(sorted(self.departements))
        return self.cache_dir / f"census_fr_{deps}_{self.year}.parquet"


def _income_bracket(median_income_eur: float) -> str:
    if np.isnan(median_income_eur):
        return "median"
    if median_income_eur < 22_000:
        return "lt22k"
    if median_income_eur < 40_000:
        return "22k_40k"
    if median_income_eur < 60_000:
        return "40k_60k"
    return "gt60k"
