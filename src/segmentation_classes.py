"""
Contains the classes used for epithelial segmentation
"""

import os
import zarr
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage import filters, segmentation, measure


class ApicalOutSegmenter:
    def __init__(self, zarr_path: str, mix: str, roi: str):
        self.zarr_path = zarr_path
        self.mix = mix
        self.roi = roi
        self.root = zarr.open(zarr_path, mode="a")

        self.roi_path = f"{self.mix}/apical_out/{self.roi}"

        assert (
            self.roi_path in self.root
        ), f"ROI '{self.roi}' not found in mix '{self.mix}'"

        self.img = self.root[mix]["apical_out"][roi]["raw_data"][:]

    def segment(self):
        img = self.img
        mask = np.invert(img == 0)
        mask = rgb2gray(mask).astype(bool)

        mask_dataset_path = f"{self.mix}/apical_out/{self.roi}/segmentation/mask"

        if mask_dataset_path in self.root:
            del self.root[mask_dataset_path]

        mask_dataset = self.root.create_dataset(mask_dataset_path, data=mask)
        mask_dataset.attrs.update(
            {"author": "Turku BioImaging", "description": "Apical-out mask"}
        )


class ApicalInSegmenter:
    def __init__(self, zarr_path: str, mix: str, roi: str):
        self.zarr_path = zarr_path
        self.mix = mix
        self.roi = roi
        self.root = zarr.open(zarr_path, mode="a")

        self.roi_path = f"{self.mix}/apical_in/{self.roi}"

        assert (
            self.roi_path in self.root
        ), f"ROI '{self.roi}' not found in mix '{self.mix}'"

        self.img = self.root[mix]["apical_in"][roi]["raw_data"][:]

    def segment(self):
        img = self.img

        # Get the outer mask
        outer_mask = np.invert(img == 0)
        outer_mask = rgb2gray(outer_mask)

        # Invert the image, apply a gaussian filter, and use adaptive threshold
        inverted_img = rgb2gray(img)
        inverted_img = (filters.gaussian(inverted_img, sigma=3) * 65535).astype(
            np.uint16
        )

        thr = filters.threshold_local(inverted_img, block_size=355)
        inverted_img = inverted_img > thr

        # Get the largest "hole".
        # This usually corresponds to the luminal space.
        holes = np.invert(inverted_img)
        holes = segmentation.clear_border(holes)
        hole_img = np.zeros_like(holes)
        hole_labels = measure.label(holes)
        regions = measure.regionprops(hole_labels)
        if regions:
            largest_hole = max(regions, key=lambda region: region.area)
            hole_img[largest_hole.coords[:, 0], largest_hole.coords[:, 1]] = 1
            hole_img = ndi.binary_fill_holes(hole_img)

        # Now take the epithelium masks, fill all holes, and then remove the largest hole
        mask = outer_mask + inverted_img
        mask = ndi.binary_fill_holes(mask)
        mask[hole_img] = 0

        # Save outputs to zarr datasets
        inv_dataset_path = (
            f"{self.mix}/apical_in/{self.roi}/segmentation/primitive_mask"
        )
        hole_dataset_path = f"{self.mix}/apical_in/{self.roi}/segmentation/largest_hole"
        mask_dataset_path = f"{self.mix}/apical_in/{self.roi}/segmentation/mask"

        if f"{self.mix}/apical_in/{self.roi}/segmentation/inverted" in self.root:
            del self.root[f"{self.mix}/apical_in/{self.roi}/segmentation/inverted"]

        if inv_dataset_path in self.root:
            del self.root[inv_dataset_path]

        if hole_dataset_path in self.root:
            del self.root[hole_dataset_path]

        if mask_dataset_path in self.root:
            del self.root[mask_dataset_path]

        inv_dataset = self.root.create_dataset(inv_dataset_path, data=inverted_img)
        inv_dataset.attrs.update(
            {"author": "Turku BioImaging", "description": "Primitive mask"}
        )

        hole_dataset = self.root.create_dataset(hole_dataset_path, data=hole_img)
        hole_dataset.attrs.update(
            {
                "author": "Turku BioImaging",
                "description": "Mask of the largest inner hole, usually corresponding to the luminal space",
            }
        )

        mask_dataset = self.root.create_dataset(mask_dataset_path, data=mask)
        mask_dataset.attrs.update(
            {"author": "Turku BioImaging", "description": "Final apical-in mask"}
        )
