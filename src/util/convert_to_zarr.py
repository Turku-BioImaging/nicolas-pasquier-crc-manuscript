"""
Convert the ROI data to zarr format, organized into a suitable heirarchy.
"""

import os
from glob import glob

import zarr
from skimage import io
from tqdm import tqdm

ROI_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "rois")
ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "zarr_data", "roi_data.zarr"
)

MIX_PROTEINS = {"mix_1": "HER2 / SORLA", "mix_2": "HER2 / HER3"}

if __name__ == "__main__":

    root = zarr.open(ZARR_PATH, mode="w")

    mix_dirs = [
        d
        for d in os.listdir(ROI_DIR)
        if (os.path.isdir(os.path.join(ROI_DIR, d)) and d != "manual")
    ]

    for mix in mix_dirs:
        mix_group = root.create_group(mix)
        mix_group.attrs["name"] = mix
        mix_group.attrs["targets"] = MIX_PROTEINS[mix]
        mix_group.attrs["author"] = (
            "Nicolas Pasquier, Cell Adhesion and Cancer Lab, University of Turku"
        )

        rois = glob(os.path.join(ROI_DIR, mix, "*.tif"))
        apical_in_rois = [r for r in rois if "_in_" in r]
        apical_out_rois = [r for r in rois if "_out_" in r]

        assert len(apical_in_rois) + len(apical_out_rois) == len(rois)

        for roi in tqdm(rois, desc=f"Processing {mix} apical in"):
            roi_name = os.path.basename(roi).split(".")[0]
            apical_class = "apical_in" if "_in_" in roi_name else "apical_out"

            img = io.imread(roi)

            r_dataset = mix_group.create_dataset(
                f"{apical_class}/{roi_name}/raw_data", data=img
            )
            r_dataset.attrs["name"] = roi_name
            r_dataset.attrs["author"] = mix_group.attrs["author"]
            r_dataset.attrs["resolution"] = {
                "unit": "microns / pixel",
                "x": 0.3250000,
                "y": 0.3250000,
            }
            r_dataset.attrs["bit_depth"] = 16
            r_dataset.attrs["dim_order"] = "YXC"
            r_dataset.attrs["dimensions"] = {
                "height": img.shape[0],
                "width": img.shape[1],
                "channels": ["DAPI", "Cy3", "AF647"],
            }
            r_dataset.attrs["height"] = img.shape[0]
            r_dataset.attrs["width"] = img.shape[1]
