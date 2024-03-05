"""
Class definitions for zone segmentation
"""

import numpy as np
import zarr
from skimage.morphology import erosion, dilation, disk

INNER_ZONE_THICKNESS = 45
OUTER_ZONE_THICKNESS = 45


def _generate_outer_zone(
    mask: np.ndarray, largest_hole_mask: np.ndarray = None
) -> np.ndarray:

    if largest_hole_mask is not None:
        mask = mask + largest_hole_mask

    # outer_zone_mask = mask + largest_hole_mask
    outer_zone_mask = np.invert(erosion(mask, disk(OUTER_ZONE_THICKNESS)))
    outer_zone_mask = np.logical_and(outer_zone_mask, mask)
    return outer_zone_mask


class ApicalOutZoner:
    def __init__(self, zarr_path: str, mix: str, roi: str):
        self.zarr_path = zarr_path
        self.mix = mix
        self.roi = roi
        self.root = zarr.open(zarr_path, mode="a")

        self.roi_path = f"{self.mix}/apical_out/{self.roi}"

        assert "_out_" in self.roi, f"ROI '{self.roi}' is not apical-out"
        assert (
            self.roi_path in self.root
        ), f"ROI '{self.roi}' not found in mix '{self.mix}'"

        self.mask = self.root[mix]["apical_out"][roi]["segmentation"]["mask"][:]

    def generate(self):
        mask = self.mask
        # outer_zone_mask = np.invert(erosion(mask, disk(OUTER_ZONE_THICKNESS)))
        outer_zone_mask = _generate_outer_zone(mask)

        outer_zone_dataset_path = (
            f"{self.mix}/apical_out/{self.roi}/segmentation/zones/outer"
        )

        if outer_zone_dataset_path in self.root:
            del self.root[outer_zone_dataset_path]

        outer_zone_dataset = self.root.create_dataset(
            outer_zone_dataset_path, data=outer_zone_mask
        )
        outer_zone_dataset.attrs.update(
            {
                "author": "Turku BioImaging",
                "description": "Apical-out outer zone",
                "mix": self.mix,
                "roi": self.roi,
                "zone_thickness": OUTER_ZONE_THICKNESS,
            }
        )


class ApicalInZoner:
    def __init__(self, zarr_path: str, mix: str, roi: str):
        self.zarr_path = zarr_path
        self.mix = mix
        self.roi = roi
        self.root = zarr.open(zarr_path, mode="a")

        self.roi_path = f"{self.mix}/apical_in/{self.roi}"

        assert "_in_" in self.roi, f"ROI '{self.roi}' is not apical-in"
        assert (
            self.roi_path in self.root
        ), f"ROI '{self.roi}' not found in mix '{self.mix}'"

        self.mask = self.root[mix]["apical_in"][roi]["segmentation"]["mask"][:]
        self.largest_hole_mask = self.root[mix]["apical_in"][roi]["segmentation"][
            "largest_hole"
        ][:]

    def generate(self, save_overlays: bool = True):

        # create outer zone
        outer_zone = _generate_outer_zone(self.mask, self.largest_hole_mask)

        # create inner zone
        inner_zone = dilation(self.largest_hole_mask, disk(INNER_ZONE_THICKNESS))
        inner_zone = np.logical_and(inner_zone, self.mask)

        # remove overlaps between outer and inner zones
        overlap_mask = np.logical_and(outer_zone, inner_zone)
        inner_zone[overlap_mask] = False
        outer_zone[overlap_mask] = False

        # clean existing datasets and create new ones
        outer_zone_dataset_path = (
            f"{self.mix}/apical_in/{self.roi}/segmentation/zones/outer"
        )
        if outer_zone_dataset_path in self.root:
            del self.root[outer_zone_dataset_path]

        outer_zone_dataset = self.root.create_dataset(
            outer_zone_dataset_path, data=outer_zone
        )
        outer_zone_dataset.attrs.update(
            {
                "author": "Turku BioImaging",
                "description": "Apical-in outer zone",
                "mix": self.mix,
                "roi": self.roi,
                "zone_thickness": OUTER_ZONE_THICKNESS,
            }
        )

        inner_zone_dataset_path = (
            f"{self.mix}/apical_in/{self.roi}/segmentation/zones/inner"
        )

        if inner_zone_dataset_path in self.root:
            del self.root[inner_zone_dataset_path]

        inner_zone_dataset = self.root.create_dataset(
            inner_zone_dataset_path, data=inner_zone
        )
        inner_zone_dataset.attrs.update(
            {
                "author": "Turku BioImaging",
                "description": "Apical-in inner zone",
                "mix": self.mix,
                "roi": self.roi,
                "zone_thickness": INNER_ZONE_THICKNESS,
            }
        )
