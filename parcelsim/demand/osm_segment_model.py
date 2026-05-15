from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.strtree import STRtree

if TYPE_CHECKING:
    from parcelsim.city import City

# Constants from Hörl & Benki (2025) / AMZ_Project_Picasso calibration
_WALKING_SPEED  = 1.4    # m/s
_HANDOVER_TIME  = 60.0   # seconds per stop
_INDOORS_TIME   = 300.0  # seconds for largest building; scales by area ratio

_RESIDENTIAL_TAGS  = {'house', 'residential', 'apartments', 'detached',
                      'terrace', 'semidetached_house', 'dormitory'}
_COMMERCIAL_TAGS   = {'commercial', 'retail', 'office', 'supermarket',
                      'mall', 'shop', 'train_station', 'hotel'}
_INDUSTRIAL_TAGS   = {'industrial', 'warehouse', 'factory', 'storage'}
_UNIVERSITY_TAGS   = {'university', 'college', 'school'}


class OSMSegmentDemandModel:
    """
    Computes per-street-segment delivery demand from OSM building data.

    For each road segment, estimates the expected delivery service time based on:
    - Number of buildings assigned to that segment
    - Building area (proxy for delivery complexity)
    - Distance to nearest parking

    Calibrated to the AMZ_Project_Picasso methodology (Hörl & Benki, 2025).

    Usage::

        from parcelsim.demand.osm_segment_model import OSMSegmentDemandModel

        model = OSMSegmentDemandModel()
        segments = model.compute(city)   # GeoDataFrame with service_time per segment
    """

    def __init__(
        self,
        buffer_m: float = 30.0,
        walking_speed: float = _WALKING_SPEED,
        handover_time: float = _HANDOVER_TIME,
        indoors_time: float = _INDOORS_TIME,
        cache_dir: Path | str = "./parcelsim_cache",
    ) -> None:
        self.buffer_m       = buffer_m
        self.walking_speed  = walking_speed
        self.handover_time  = handover_time
        self.indoors_time   = indoors_time
        self.cache_dir      = Path(cache_dir)

    def compute(self, city: "City") -> gpd.GeoDataFrame:
        """
        Returns a GeoDataFrame with one row per road segment and columns:
        segment_id, geometry, highway, is_oneway, segment_length,
        building_count, land_use, parking_dist, service_time.
        """
        cache_path = self._cache_path(city)
        if cache_path.exists():
            return gpd.read_parquet(cache_path)

        import osmnx as ox

        study_poly = city.study_area.to_crs("EPSG:4326").union_all()

        print("Downloading road network from OSM...")
        G = ox.graph_from_polygon(study_poly, network_type="drive", simplify=True)
        _, edges = ox.graph_to_gdfs(G)
        edges = edges.reset_index()

        print("Downloading buildings from OSM...")
        try:
            buildings = ox.features_from_polygon(study_poly, tags={"building": True})
            buildings = buildings[buildings.geometry.geom_type.isin(
                ["Polygon", "MultiPolygon"]
            )].copy()
        except Exception:
            buildings = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        print("Downloading parking from OSM...")
        try:
            parking = ox.features_from_polygon(
                study_poly, tags={"amenity": "parking"}
            )
            parking = parking[parking.geometry.notnull()].copy()
            parking_pts = parking.geometry.centroid
        except Exception:
            parking_pts = gpd.GeoSeries([], crs="EPSG:4326")

        print("Computing segment features...")
        result = self._build_segment_features(edges, buildings, parking_pts, city.crs)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path)
        return result

    # ------------------------------------------------------------------ #
    # Core computation                                                     #
    # ------------------------------------------------------------------ #

    def _build_segment_features(
        self,
        edges: gpd.GeoDataFrame,
        buildings: gpd.GeoDataFrame,
        parking_pts: gpd.GeoSeries,
        crs: str,
    ) -> gpd.GeoDataFrame:
        edges_proj = edges.to_crs(crs).copy()
        edges_proj["segment_id"] = (
            edges_proj["u"].astype(str) + "_"
            + edges_proj["v"].astype(str) + "_"
            + edges_proj["key"].astype(str)
        )
        edges_proj["segment_length"] = edges_proj.geometry.length
        edges_proj["is_oneway"] = edges_proj.get("oneway", False).fillna(False)

        # ── Assign buildings to nearest segment ──────────────────────────
        if len(buildings) > 0:
            bld_proj = buildings.to_crs(crs).copy()
            bld_proj["area_m2"] = bld_proj.geometry.area
            bld_proj["bld_type"] = bld_proj.get("building", "yes").fillna("yes").astype(str)
            bld_proj["centroid"] = bld_proj.geometry.centroid

            # Spatial index on segment buffers
            seg_buf = edges_proj.geometry.buffer(self.buffer_m)
            tree = STRtree(seg_buf.values)

            max_area = bld_proj["area_m2"].replace(0, np.nan).max()
            if np.isnan(max_area) or max_area == 0:
                max_area = 1.0

            # Parking distance per building
            if len(parking_pts) > 0:
                park_proj = parking_pts.to_crs(crs)
                park_tree = STRtree(park_proj.values)
                def _park_dist(pt):
                    idx = park_tree.nearest(pt)
                    return pt.distance(park_proj.iloc[idx])
            else:
                def _park_dist(pt):
                    return 15.0  # default: 15 m if no parking data

            # Accumulate per segment
            seg_building_count  = np.zeros(len(edges_proj), dtype=int)
            seg_service_time    = np.zeros(len(edges_proj), dtype=float)
            seg_land_use_votes  = [[] for _ in range(len(edges_proj))]

            for _, bld in bld_proj.iterrows():
                pt    = bld["centroid"]
                area  = bld["area_m2"]
                btype = bld["bld_type"]

                # Find segments within buffer_m
                hits = tree.query(pt)
                if len(hits) == 0:
                    continue

                # Assign to the closest segment
                dists = [pt.distance(edges_proj.geometry.iloc[i]) for i in hits]
                seg_idx = hits[int(np.argmin(dists))]

                park_d = _park_dist(pt)
                st = self._service_time(area, park_d, max_area)

                seg_building_count[seg_idx] += 1
                seg_service_time[seg_idx]   += st
                seg_land_use_votes[seg_idx].append(btype)

            edges_proj["building_count"] = seg_building_count
            edges_proj["service_time"]   = seg_service_time
            edges_proj["parking_dist"]   = edges_proj.geometry.apply(
                lambda g: _park_dist(g.centroid) if seg_building_count[edges_proj.index.get_loc(g.name) if hasattr(g, 'name') else 0] > 0 else np.nan
            )
            edges_proj["land_use"] = [
                _majority_land_use(votes) for votes in seg_land_use_votes
            ]
        else:
            edges_proj["building_count"] = 0
            edges_proj["service_time"]   = 0.0
            edges_proj["parking_dist"]   = np.nan
            edges_proj["land_use"]       = "unknown"

        keep = ["segment_id", "geometry", "highway", "is_oneway",
                "segment_length", "building_count", "land_use",
                "parking_dist", "service_time"]
        result = edges_proj[[c for c in keep if c in edges_proj.columns]].copy()
        return result.reset_index(drop=True)

    def _service_time(self, area: float, dist_parking: float, max_area: float) -> float:
        walking   = 2 * dist_parking / self.walking_speed
        if np.isnan(area) or area <= 0:
            indoors = self.indoors_time * 0.5
        else:
            indoors = self.indoors_time * area / max_area
        return walking + indoors + self.handover_time

    def _cache_path(self, city: "City") -> Path:
        key = hashlib.md5(
            f"{city.name}_{self.buffer_m}_{self.walking_speed}".encode()
        ).hexdigest()[:10]
        return self.cache_dir / f"osm_segments_{city.name}_{key}.parquet"

    def summary(self, segments: gpd.GeoDataFrame) -> str:
        lines = [
            f"OSMSegmentDemand [{segments['segment_id'].nunique()} segments]",
            f"  Total service time : {segments['service_time'].sum():>12,.0f} s",
            f"  Mean per segment   : {segments['service_time'].mean():>12.1f} s",
            f"  Segments w/ demand : {(segments['building_count'] > 0).sum():>12,}",
            f"  Total buildings    : {segments['building_count'].sum():>12,}",
        ]
        land = segments['land_use'].value_counts()
        lines.append("  Land use breakdown:")
        for lu, cnt in land.items():
            lines.append(f"    {lu:<20} {cnt:>6,} segments")
        return "\n".join(lines)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

def _majority_land_use(building_types: list[str]) -> str:
    if not building_types:
        return "unknown"
    counts = {"residential": 0, "commercial": 0, "industrial": 0,
              "university": 0, "mixed": 0}
    for bt in building_types:
        if bt in _RESIDENTIAL_TAGS:
            counts["residential"] += 1
        elif bt in _COMMERCIAL_TAGS:
            counts["commercial"] += 1
        elif bt in _INDUSTRIAL_TAGS:
            counts["industrial"] += 1
        elif bt in _UNIVERSITY_TAGS:
            counts["university"] += 1
        else:
            counts["mixed"] += 1
    return max(counts, key=counts.get)
