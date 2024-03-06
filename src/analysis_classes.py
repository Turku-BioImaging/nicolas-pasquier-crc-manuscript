"""
Classes for ROI data analysis.
"""

import os
import pandas as pd
import zarr
import numpy as np

PIXEL_SIZE = 0.325


class RoiAnalyzer:
    def __init__(self, zarr_path: str, mix: str, apical_type: str, roi: str):
        self.zarr_path = zarr_path

        assert apical_type in ["apical_in", "apical_out"], "Invalid apical type."
        self.apical_type = apical_type

        self.roi_path = f"{mix}/{apical_type}/{roi}"

        try:
            self.root = zarr.open(zarr_path, mode="r")
        except FileNotFoundError:
            raise ValueError(f"Zarr file not found at {zarr_path}")

    def analyze(self) -> dict:
        raw_data = self.root[self.roi_path]["raw_data"][:]
        cy3 = raw_data[:, :, 1]
        af647 = raw_data[:, :, 2]

        roi_mask = self.root[self.roi_path]["segmentation"]["mask"][:]
        outer_zone_mask = self.root[self.roi_path]["segmentation"]["zones"]["outer"][:]
        if self.apical_type == "apical_in":
            inner_zone_mask = self.root[self.roi_path]["segmentation"]["zones"][
                "inner"
            ][:]

        roi_pixels = np.count_nonzero(roi_mask)
        roi_area = roi_pixels * PIXEL_SIZE
        oz_pixels = np.count_nonzero(outer_zone_mask)
        oz_area = oz_pixels * PIXEL_SIZE

        if self.apical_type == "apical_in":
            iz_pixels = np.count_nonzero(inner_zone_mask)
            iz_area = iz_pixels * PIXEL_SIZE
        else:
            iz_pixels = None
            iz_area = None

        # Cy3 channel data
        cy3_id = cy3[roi_mask].sum()
        cy3_mean = cy3[roi_mask].mean()
        cy3_oz_id = cy3[outer_zone_mask].sum()
        cy3_oz_mean = cy3[outer_zone_mask].mean()

        if self.apical_type == "apical_in":
            cy3_iz_id = cy3[inner_zone_mask].sum()
            cy3_iz_mean = cy3[inner_zone_mask].mean()
        else:
            cy3_iz_id = None
            cy3_iz_mean = None

        # AF647 channel data
        af647_id = af647[roi_mask].sum()
        af647_mean = af647[roi_mask].mean()
        af647_oz_id = af647[outer_zone_mask].sum()
        af647_oz_mean = af647[outer_zone_mask].mean()

        if self.apical_type == "apical_in":
            af647_iz_id = af647[inner_zone_mask].sum()
            af647_iz_mean = af647[inner_zone_mask].mean()
        else:
            af647_iz_id = None
            af647_iz_mean = None

        return {
            "roi_name": os.path.basename(self.roi_path),
            "mix_name": os.path.basename(os.path.dirname(self.roi_path)),
            "roi_pixels": roi_pixels,
            "roi_area": roi_area,
            "oz_pixels": oz_pixels,
            "oz_area": oz_area,
            "iz_pixels": iz_pixels,
            "iz_area": iz_area,
            "cy3_id": cy3_id,
            "cy3_mean": cy3_mean,
            "cy3_oz_id": cy3_oz_id,
            "cy3_oz_mean": cy3_oz_mean,
            "cy3_iz_id": cy3_iz_id,
            "cy3_iz_mean": cy3_iz_mean,
            "af647_id": af647_id,
            "af647_mean": af647_mean,
            "af647_oz_id": af647_oz_id,
            "af647_oz_mean": af647_oz_mean,
            "af647_iz_id": af647_iz_id,
            "af647_iz_mean": af647_iz_mean,
        }
