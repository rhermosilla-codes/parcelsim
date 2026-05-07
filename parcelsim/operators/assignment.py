from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from parcelsim.city import City
    from parcelsim.demand.base import ParcelDemand
    from parcelsim.operators.operator import Operator, OperatorRegistry


@dataclass
class OperatorAssignment:
    """
    Per-zone parcel assignment to operator and depot after service area computation.

    zone_assignments GeoDataFrame columns:
        zone_id       str
        geometry      Polygon
        area_km2      float
        centroid_x    float
        centroid_y    float
        n_delivery    float    total daily parcels (all operators)
        n_pickup      float
        <operator_id>_delivery  float   parcels for this operator
        <operator_id>_depot     str     assigned depot id
        <operator_id>_dist_km   float   depot→zone centroid distance
    """

    demand: "ParcelDemand"
    registry: "OperatorRegistry"
    zone_assignments: gpd.GeoDataFrame
    metadata: dict = field(default_factory=dict)

    def for_operator(self, operator_id: str) -> gpd.GeoDataFrame:
        """Return zone rows with columns renamed for a specific operator."""
        cols = ["zone_id", "geometry", "area_km2", "centroid_x", "centroid_y",
                f"{operator_id}_delivery", f"{operator_id}_depot", f"{operator_id}_dist_km"]
        sub = self.zone_assignments[cols].copy()
        return sub.rename(columns={
            f"{operator_id}_delivery": "n_delivery",
            f"{operator_id}_depot":    "depot_id",
            f"{operator_id}_dist_km":  "dist_km",
        })

    def to_segments(
        self,
        city: "City",
        weight: str = "road_length",
    ) -> gpd.GeoDataFrame:
        """
        Disaggregate zone-level demand to street segments.

        Requires city.road_network (nx.MultiDiGraph from osmnx).
        Philipp's G_primal can be passed directly via city.road_network.

        Parameters
        ----------
        city : City
            City with road_network set (any osmnx MultiDiGraph in the same CRS).
        weight : str
            "road_length" — prorate by segment length within zone (default)
            "uniform"     — equal split across all segments in zone

        Returns
        -------
        GeoDataFrame indexed by (u, v, key) — same index as osmnx edges:
            osmid           OSM edge ID
            geometry        LineString (projected CRS)
            zone_id         parent zone
            highway         road type
            seg_weight      fractional share of zone demand for this segment
            n_delivery      expected parcels/day
            {op}_delivery   parcels/day per operator
            {op}_depot      assigned depot ID (inherited from zone)
        """
        if city.road_network is None:
            raise ValueError(
                "city.road_network is required. Set city.road_network = G_primal."
            )
        try:
            import osmnx as ox
        except ImportError:
            raise ImportError("osmnx is required: pip install osmnx")

        _, edges = ox.graph_to_gdfs(city.road_network)
        edges = edges.reset_index().to_crs(city.crs)

        # Spatial join: assign each segment to a zone via centroid
        centroids = edges.copy()
        centroids["geometry"] = edges.geometry.centroid
        zones = self.zone_assignments[["zone_id", "geometry"]].copy()
        joined = gpd.sjoin(centroids, zones, how="left", predicate="within")
        edges["zone_id"] = joined["zone_id"].values

        # Compute weight of each segment within its zone
        if weight == "road_length":
            edges["_len"] = edges.geometry.length
            zone_total = edges.groupby("zone_id")["_len"].transform("sum")
            edges["seg_weight"] = (edges["_len"] / zone_total.replace(0, 1)).fillna(0)
            edges = edges.drop(columns=["_len"])
        else:
            zone_count = edges.groupby("zone_id")["zone_id"].transform("count")
            edges["seg_weight"] = (1.0 / zone_count).fillna(0)

        # Prorate demand from zone to segment
        za = self.zone_assignments.set_index("zone_id")

        def _prorate(col: str) -> pd.Series:
            return edges["zone_id"].map(za[col]).fillna(0) * edges["seg_weight"]

        edges["n_delivery"] = _prorate("n_delivery")

        # Propagate std: Poisson sub-sampling → std scales with sqrt(weight)
        if "n_delivery_std" in self.zone_assignments.columns:
            zone_std = edges["zone_id"].map(za["n_delivery_std"]).fillna(0)
            edges["n_delivery_std"] = np.sqrt(edges["seg_weight"]) * zone_std
            edges["n_delivery_p05"] = np.maximum(
                0, edges["n_delivery"] - 1.645 * edges["n_delivery_std"]
            )
            edges["n_delivery_p95"] = (
                edges["n_delivery"] + 1.645 * edges["n_delivery_std"]
            )

        for op in self.registry.operators:
            del_col = f"{op.operator_id}_delivery"
            dep_col = f"{op.operator_id}_depot"
            if del_col in self.zone_assignments.columns:
                edges[del_col] = _prorate(del_col)
            if dep_col in self.zone_assignments.columns:
                edges[dep_col] = edges["zone_id"].map(za[dep_col])

        keep = ["u", "v", "key", "osmid", "geometry", "zone_id",
                "highway", "seg_weight", "n_delivery",
                "n_delivery_std", "n_delivery_p05", "n_delivery_p95"]
        keep += [c for c in edges.columns
                 if (c.endswith("_delivery") and c != "n_delivery")
                 or c.endswith("_depot")]
        keep = [c for c in keep if c in edges.columns]
        return edges[keep].set_index(["u", "v", "key"])

    def to_blocks(
        self,
        blocks: gpd.GeoDataFrame,
        city: "City",
        weight: str = "area",
        block_id_col: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Disaggregate zone-level demand to city blocks (or any polygon units).

        Parameters
        ----------
        blocks : GeoDataFrame
            Polygon GeoDataFrame representing blocks (OSM blocks, H3, custom, etc.).
            Must have a geometry column in any CRS — will be reprojected.
        city : City
            Used for CRS reprojection.
        weight : str
            "area"    — prorate by block area within zone (default)
            "uniform" — equal split across blocks in zone
        block_id_col : str | None
            Column in blocks to use as block identifier.
            If None, uses the GeoDataFrame index.

        Returns
        -------
        GeoDataFrame indexed by block_id with columns:
            geometry, zone_id, block_weight, n_delivery, n_delivery_std,
            n_delivery_p05, n_delivery_p95, {op}_delivery, {op}_depot
        """
        blocks = blocks.to_crs(city.crs).copy()
        if block_id_col is None:
            blocks = blocks.reset_index(drop=True)
            blocks["block_id"] = blocks.index.astype(str)
            block_id_col = "block_id"

        # Spatial join: assign each block centroid to a zone
        centroids = blocks.copy()
        centroids["geometry"] = blocks.geometry.centroid
        zones = self.zone_assignments[["zone_id", "geometry"]].copy()
        joined = gpd.sjoin(centroids, zones, how="left", predicate="within")
        blocks["zone_id"] = joined["zone_id"].values

        # Compute weight within zone
        if weight == "area":
            blocks["_area"] = blocks.geometry.area
            zone_total = blocks.groupby("zone_id")["_area"].transform("sum")
            blocks["block_weight"] = (blocks["_area"] / zone_total.replace(0, 1)).fillna(0)
            blocks = blocks.drop(columns=["_area"])
        else:
            zone_count = blocks.groupby("zone_id")["zone_id"].transform("count")
            blocks["block_weight"] = (1.0 / zone_count).fillna(0)

        za = self.zone_assignments.set_index("zone_id")

        def _prorate(col: str) -> pd.Series:
            return blocks["zone_id"].map(za[col]).fillna(0) * blocks["block_weight"]

        blocks["n_delivery"] = _prorate("n_delivery")

        if "n_delivery_std" in self.zone_assignments.columns:
            zone_std = blocks["zone_id"].map(za["n_delivery_std"]).fillna(0)
            blocks["n_delivery_std"] = np.sqrt(blocks["block_weight"]) * zone_std
            blocks["n_delivery_p05"] = np.maximum(
                0, blocks["n_delivery"] - 1.645 * blocks["n_delivery_std"]
            )
            blocks["n_delivery_p95"] = (
                blocks["n_delivery"] + 1.645 * blocks["n_delivery_std"]
            )

        for op in self.registry.operators:
            del_col = f"{op.operator_id}_delivery"
            dep_col = f"{op.operator_id}_depot"
            if del_col in self.zone_assignments.columns:
                blocks[del_col] = _prorate(del_col)
            if dep_col in self.zone_assignments.columns:
                blocks[dep_col] = blocks["zone_id"].map(za[dep_col])

        keep = [block_id_col, "geometry", "zone_id", "block_weight",
                "n_delivery", "n_delivery_std", "n_delivery_p05", "n_delivery_p95"]
        keep += [c for c in blocks.columns
                 if (c.endswith("_delivery") and c != "n_delivery")
                 or c.endswith("_depot")]
        keep = [c for c in keep if c in blocks.columns]
        return blocks[keep].set_index(block_id_col)

    def summary(self) -> str:
        lines = ["OperatorAssignment"]
        for op in self.registry.operators:
            col = f"{op.operator_id}_delivery"
            if col in self.zone_assignments.columns:
                total = self.zone_assignments[col].sum()
                lines.append(f"  {op.operator_id:10s} {total:>12,.0f} parcels/day")
        return "\n".join(lines)


def assign_parcels(
    demand: "ParcelDemand",
    registry: "OperatorRegistry",
    city: "City",
    method: str = "nearest_euclidean",
) -> OperatorAssignment:
    """
    Assign zones to operators (by market share) and to depots (by distance).

    method:
      "nearest_euclidean"  — depot closest to zone centroid in projected CRS (fast)
      "nearest_network"    — shortest path on road network (requires city.road_network)
    """
    zones = demand.zone_demand.copy()
    depots_gdf = registry.all_depots(crs="EPSG:4326").to_crs(city.crs)

    for op in registry.operators:
        op_depots = depots_gdf[depots_gdf["operator_id"] == op.operator_id].copy()
        if op_depots.empty:
            zones[f"{op.operator_id}_delivery"] = zones["n_delivery"] * op.market_share
            zones[f"{op.operator_id}_pickup"] = zones["n_pickup"] * op.market_share
            zones[f"{op.operator_id}_depot"] = None
            zones[f"{op.operator_id}_dist_km"] = np.nan
            continue

        if method == "nearest_euclidean":
            depot_id_col, dist_col = _assign_euclidean(zones, op_depots)
        elif method == "nearest_network":
            depot_id_col, dist_col = _assign_network(zones, op_depots, city)
        else:
            raise ValueError(f"Unknown assignment method: {method}")

        zones[f"{op.operator_id}_delivery"] = zones["n_delivery"] * op.market_share
        zones[f"{op.operator_id}_pickup"] = zones["n_pickup"] * op.market_share
        zones[f"{op.operator_id}_depot"] = depot_id_col
        zones[f"{op.operator_id}_dist_km"] = dist_col

    return OperatorAssignment(
        demand=demand,
        registry=registry,
        zone_assignments=zones,
        metadata={"method": method},
    )


def _assign_euclidean(
    zones: gpd.GeoDataFrame,
    depots: gpd.GeoDataFrame,
) -> tuple[pd.Series, pd.Series]:
    cx = zones["centroid_x"].values
    cy = zones["centroid_y"].values
    dx = np.array([d.x for d in depots.geometry])
    dy = np.array([d.y for d in depots.geometry])
    depot_ids = depots["depot_id"].values

    # Vectorized nearest-depot: shape (n_zones, n_depots)
    dist_matrix = np.sqrt(
        (cx[:, None] - dx[None, :]) ** 2 + (cy[:, None] - dy[None, :]) ** 2
    )
    nearest_idx = dist_matrix.argmin(axis=1)
    nearest_depot = pd.Series(depot_ids[nearest_idx], index=zones.index)
    nearest_dist  = pd.Series(dist_matrix[np.arange(len(zones)), nearest_idx] / 1000,
                               index=zones.index)
    return nearest_depot, nearest_dist


def _assign_network(
    zones: gpd.GeoDataFrame,
    depots: gpd.GeoDataFrame,
    city: "City",
) -> tuple[pd.Series, pd.Series]:
    if city.road_network is None:
        raise ValueError(
            "city.road_network is required for 'nearest_network' assignment. "
            "Load with City.from_osmnx(..., load_network=True)."
        )
    import networkx as nx
    import osmnx as ox

    G = city.road_network
    depot_ids = depots["depot_id"].values
    depot_nodes = [
        ox.nearest_nodes(G, X=d.x, Y=d.y) for d in depots.geometry
    ]
    zone_nodes = [
        ox.nearest_nodes(G, X=row["centroid_x"], Y=row["centroid_y"])
        for _, row in zones.iterrows()
    ]

    nearest_depot = []
    nearest_dist = []
    for z_node in zone_nodes:
        best_depot, best_dist = None, float("inf")
        for depot_id, d_node in zip(depot_ids, depot_nodes):
            try:
                dist = nx.shortest_path_length(G, d_node, z_node, weight="length")
            except nx.NetworkXNoPath:
                dist = float("inf")
            if dist < best_dist:
                best_dist, best_depot = dist, depot_id
        nearest_depot.append(best_depot)
        nearest_dist.append(best_dist / 1000)

    return (
        pd.Series(nearest_depot, index=zones.index),
        pd.Series(nearest_dist, index=zones.index),
    )
