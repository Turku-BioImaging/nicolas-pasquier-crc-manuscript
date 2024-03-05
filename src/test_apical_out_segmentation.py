"""
Test script for segmenting 'apical-out' ROIs.
"""

import os
import numpy as np
import zarr
from tqdm import tqdm
from skimage.color import rgb2gray

ZARR_PATH = os.path.join(os.path.dirname(__file__), "..", "zarr_data", "roi_data.zarr")
ATTRS = {"author": "Turku BioImaging"}


if __name__ == "__main__":
    root = zarr.open(ZARR_PATH, mode="a")
    mixes = list(root.keys())

    for mix in mixes:
        rois = list(root[mix]["apical_out"].keys())

        for roi in tqdm(rois, desc=f"Processing {mix} apical_out ROIs"):
            img = root[mix]["apical_out"][roi]["raw_data"][:]

            # get the outer mask
            mask = np.invert(img == 0)
            mask = rgb2gray(mask).astype(bool)

            mask_dataset_path = f"apical_out/{roi}/segmentation/mask"
            mask_dataset = root[mix].create_dataset(mask_dataset_path, data=mask)
            mask_dataset.attrs.update(ATTRS)
