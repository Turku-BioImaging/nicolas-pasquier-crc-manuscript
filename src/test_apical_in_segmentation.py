"""
This script is a test for apical segmentation.
"""
import os

import numpy as np
import zarr
from scipy import ndimage as ndi
from skimage import filters, measure, segmentation
from skimage.color import rgb2gray
from tqdm import tqdm

ZARR_PATH = os.path.join(os.path.dirname(__file__), "..", "zarr_data", "roi_data.zarr")
ATTRS = {"author": "Turku BioImaging"}

if __name__ == "__main__":
    root = zarr.open(ZARR_PATH, mode="a")
    mixes = list(root.keys())

    for mix in mixes:
        rois = list(root[mix]["apical_in"].keys())

        for roi in tqdm(rois, desc=f"Processing {mix} apical_in ROIs"):
            img = root[mix]["apical_in"][roi]["raw_data"][:]

            # get the outer mask
            outer_mask = np.invert(img == 0)
            outer_mask = rgb2gray(outer_mask)

            # inver the image, apply a gaussian filter, and threshold using Otsu
            inverted_img = rgb2gray(img)
            inverted_img = (filters.gaussian(inverted_img, sigma=3) * 65535).astype(
                np.uint16
            )

            # thr = filters.threshold_otsu(inverted_img)
            thr = filters.threshold_local(inverted_img, block_size=355)
            inverted_img = inverted_img > thr

            inv_dataset_path = f"apical_in/{roi}/segmentation/inverted"
            if inv_dataset_path in root[mix]:
                del root[mix][inv_dataset_path]

            inv_dataset = root[mix].create_dataset(
                f"apical_in/{roi}/segmentation/inverted", data=inverted_img
            )
            inv_dataset.attrs.update(ATTRS)

            # get the largest "hole"
            holes = np.invert(inverted_img)
            holes = segmentation.clear_border(holes)

            hole_img = np.zeros_like(holes)

            hole_labels = measure.label(holes)
            regions = measure.regionprops(hole_labels)
            if regions:
                largest_hole = max(regions, key=lambda region: region.area)

                hole_img[largest_hole.coords[:, 0], largest_hole.coords[:, 1]] = 1
                hole_img = ndi.binary_fill_holes(hole_img)

            hole_dataset_path = f"apical_in/{roi}/segmentation/largest_hole"
            if hole_dataset_path in root[mix]:
                del root[mix][hole_dataset_path]

            hole_dataset = root[mix].create_dataset(hole_dataset_path, data=hole_img)
            hole_dataset.attrs.update(ATTRS)

            # now take the mask, fill all holes, and then remove the largest hole
            mask = outer_mask + inverted_img
            mask = ndi.binary_fill_holes(mask)
            mask[hole_img] = 0

            mask_dataset_path = f"apical_in/{roi}/segmentation/mask"
            if mask_dataset_path in root[mix]:
                del root[mix][mask_dataset_path]

            mask_dataset = root[mix].create_dataset(mask_dataset_path, data=mask)
            mask_dataset.attrs.update(ATTRS)
