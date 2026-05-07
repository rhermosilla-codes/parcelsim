from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import yaml
from shapely.geometry import Point


@dataclass
class Depot:
    depot_id: str
    operator_id: str
    location: Point
    max_vehicles: int | None = None
    properties: dict = field(default_factory=dict)

    def as_geodataframe(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            [{"depot_id": self.depot_id, "operator_id": self.operator_id,
              "geometry": self.location}],
            geometry="geometry",
            crs=crs,
        )


@dataclass
class Operator:
    operator_id: str
    name: str
    market_share: float
    depots: list[Depot]
    delivery_threshold: float = 0.7
    pickup_threshold: float = 0.66

    def __post_init__(self) -> None:
        if not (0 < self.market_share <= 1):
            raise ValueError(f"market_share must be in (0, 1], got {self.market_share}")


class OperatorRegistry:
    """Collection of parcel operators with market shares that sum to 1."""

    def __init__(self, operators: list[Operator]) -> None:
        self.operators = operators
        total = sum(o.market_share for o in operators)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Operator market shares sum to {total:.3f}, expected 1.0")

    @classmethod
    def from_builtin(cls, preset: str) -> "OperatorRegistry":
        """
        Load a built-in operator preset.

        Available presets:
          "us_2021"  — USPS, UPS, FedEx, Amazon (NYC depot locations, Yang et al. 2024)
          "fr_2024"  — La Poste, Chronopost, UPS, DPD (Lyon depots, Hörl et al. 2025)
        """
        presets_dir = Path(__file__).parent / "builtin"
        yaml_path = presets_dir / f"{preset}.yaml"
        if not yaml_path.exists():
            available = [p.stem for p in presets_dir.glob("*.yaml")]
            raise FileNotFoundError(
                f"Preset '{preset}' not found. Available: {available}"
            )
        return cls.from_yaml(yaml_path)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "OperatorRegistry":
        with open(path) as f:
            data = yaml.safe_load(f)
        operators = [_operator_from_dict(d) for d in data["operators"]]
        return cls(operators)

    def get(self, operator_id: str) -> Operator:
        for op in self.operators:
            if op.operator_id == operator_id:
                return op
        raise KeyError(f"Operator '{operator_id}' not found")

    def all_depots(self, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        rows = []
        for op in self.operators:
            for depot in op.depots:
                rows.append({
                    "depot_id": depot.depot_id,
                    "operator_id": op.operator_id,
                    "operator_name": op.name,
                    "geometry": depot.location,
                    "max_vehicles": depot.max_vehicles,
                })
        if not rows:
            return gpd.GeoDataFrame(columns=["depot_id", "operator_id", "geometry"],
                                    geometry="geometry", crs=crs)
        return gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)

    def __repr__(self) -> str:
        parts = [f"{o.operator_id}({o.market_share:.0%})" for o in self.operators]
        return f"OperatorRegistry([{', '.join(parts)}])"


def _operator_from_dict(d: dict[str, Any]) -> Operator:
    depots = [
        Depot(
            depot_id=dep["id"],
            operator_id=d["id"],
            location=Point(dep["lon"], dep["lat"]),
            max_vehicles=dep.get("max_vehicles"),
            properties=dep.get("properties", {}),
        )
        for dep in d.get("depots", [])
    ]
    return Operator(
        operator_id=d["id"],
        name=d["name"],
        market_share=d["market_share"],
        depots=depots,
        delivery_threshold=d.get("delivery_threshold", 0.7),
        pickup_threshold=d.get("pickup_threshold", 0.66),
    )
