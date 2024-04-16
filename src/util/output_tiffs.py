"""
Utility script for generating tiff outputs of ROI roi data and their generated masks.
"""

import os
import numpy as np
import zarr
from skimage import io, img_as_ubyte

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "zarr_data", "roi_data.zarr"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tiff_outputs")


if __name__ == "__main__":
    root = zarr.open(ZARR_PATH, mode="r")

    mixes = list(root.keys())
    for mix in mixes:

        for ap in list(root[mix].keys()):

            roi_keys = list(root[mix][ap].keys())

            for roi_key in roi_keys:
                raw_data = root[mix][ap][roi_key]["raw_data"]
                mask = root[mix][ap][roi_key]["segmentation"]["mask"]
                # zone_overlay = root[mix][ap][roi_key]["segmentation"]["zones"][
                #     "overlay"
                # ]

                roi_dir = os.path.join(OUTPUT_DIR, mix, ap, roi_key)
                os.makedirs(roi_dir, exist_ok=True)

                io.imsave(
                    os.path.join(roi_dir, "raw_data.tif"),
                    raw_data,
                    check_contrast=False,
                )
                io.imsave(
                    os.path.join(roi_dir, "mask.tif"),
                    img_as_ubyte(mask),
                    check_contrast=False,
                )
                # io.imsave(
                #     os.path.join(roi_dir, "zone_overlay.tif"),
                #     zone_overlay,
                #     check_contrast=False,
                # )
