"""
Code to measure correlation between different signals in ROIs.

Correlation analysis of intensity HER2 vs HER3 and HER2 vs SORLA within on patient samples for 1) whole region, (apical-in and apical-out), and for 2) basal and apical pole zones separately. Plot the correlations for each individual ROI and for average per patient.
"""

import os
import zarr

# from joblib import Parallel, delayed
import numpy as np
import pandas as pd

# from skimage import io, img_as_ubyte
from scipy.stats import pearsonr
from tqdm import tqdm

ZARR_PATH = os.path.join(os.path.dirname(__file__), "..", "zarr_data", "roi_data.zarr")

MIX_1_CH_NAMES = ["HER2", "SORLA"]
MIX_2_CH_NAMES = ["HER2", "HER3"]

root = zarr.open(ZARR_PATH, mode="r")
mixes = list(root.keys())
rois = [
    (mix, apical_type, roi)
    for mix in mixes
    for apical_type in list(root[mix].keys())
    for roi in list(root[mix][apical_type].keys())
]

# process Mix 1 data
# HER2 -> Ch2
# SORLA -> Ch3

mix_1_data = []
mix_2_data = []

# apical-in
for item in tqdm(rois):
    mix, apical_type, roi = item

    if apical_type == "apical_out":
        continue

    # process apical_in types
    roi_path = root[f"{mix}/{apical_type}/{roi}"]

    raw_data = roi_path["raw_data"]
    roi_mask = roi_path["segmentation"]["mask"]
    apical_mask = roi_path["segmentation"]["zones"].get(
        "inner_manual", roi_path["segmentation"]["zones"]["inner"]
    )
    basal_mask = roi_path["segmentation"]["zones"]["outer"]

    # roi_mask = img_as_ubyte(roi_mask)
    # apical_mask = img_as_ubyte(apical_mask)
    # basal_mask = img_as_ubyte(basal_mask)

    ch2_roi = raw_data[:, :, 1][roi_mask]
    # ch2_roi[roi_mask == 0] = 0

    ch3_roi = raw_data[:, :, 2][roi_mask]
    # ch3_roi[roi_mask == 0] = 0

    ch2_apical = raw_data[:, :, 1][apical_mask]
    # ch2_apical[apical_mask == 0] = 0

    ch2_basal = raw_data[:, :, 1][basal_mask]
    # ch2_basal[basal_mask == 0] = 0

    ch3_apical = raw_data[:, :, 2][apical_mask]
    # ch3_apical[apical_mask == 0] = 0

    ch3_basal = raw_data[:, :, 2][basal_mask]
    # ch3_basal[basal_mask == 0] = 0

    if mix == "mix_1":
        ch2_name, ch3_name = MIX_1_CH_NAMES
    elif mix == "mix_2":
        ch2_name, ch3_name = MIX_2_CH_NAMES
    else:
        raise ValueError(f"Unknown mix: {mix}")

    # data for whole roi
    whole_roi_data = {
        "roi": roi,
        "mix": mix,
        "roi_type": apical_type,
        "zone": "whole_roi",
        f"{ch2_name}_total_intensity": ch2_roi.sum(),
        f"{ch3_name}_total_intensity": ch3_roi.sum(),
        f"{ch2_name}_{ch3_name}_correlation": (
            pearsonr(ch2_roi, ch3_roi)[0] if len(ch2_roi) > 0 else 0
        ),
    }

    # data for apical zone

    apical_data = {
        "roi": roi,
        "mix": mix,
        "roi_type": apical_type,
        "zone": "apical",
        f"{ch2_name}_total_intensity": ch2_apical.sum(),
        f"{ch3_name}_total_intensity": ch3_apical.sum(),
        f"{ch2_name}_{ch3_name}_correlation": (
            pearsonr(ch2_apical, ch3_apical)[0] if len(ch2_apical) > 0 else 0
        ),
    }

    # data for basal zone
    basal_data = {
        "roi": roi,
        "mix": mix,
        "roi_type": apical_type,
        "zone": "basal",
        f"{ch2_name}_total_intensity": ch2_basal.sum(),
        f"{ch3_name}_total_intensity": ch3_basal.sum(),
        f"{ch2_name}_{ch3_name}_correlation": (
            pearsonr(ch2_basal, ch3_basal)[0] if len(ch2_basal) > 0 else 0
        ),
    }

    if mix == "mix_1":
        mix_1_data.extend([whole_roi_data, apical_data, basal_data])
    elif mix == "mix_2":
        mix_2_data.extend([whole_roi_data, apical_data, basal_data])

# apical-out
for item in tqdm(rois):
    mix, apical_type, roi = item

    if apical_type == "apical_in":
        continue

    roi_path = root[f"{mix}/{apical_type}/{roi}"]
    raw_data = roi_path["raw_data"]
    roi_mask = roi_path["segmentation"]["mask"]
    apical_mask = roi_path["segmentation"]["zones"]["outer"]

    # roi_mask = img_as_ubyte(roi_mask)
    # apical_mask = img_as_ubyte(apical_mask)

    ch2_roi = raw_data[:, :, 1][roi_mask]
    # ch2_roi[roi_mask == 0] = 0

    ch3_roi = raw_data[:, :, 2][roi_mask]
    # ch3_roi[roi_mask == 0] = 0

    ch2_apical = raw_data[:, :, 1][apical_mask]
    # ch2_apical[apical_mask == 0] = 0

    ch3_apical = raw_data[:, :, 2][apical_mask]
    # ch3_apical[apical_mask == 0] = 0

    if mix == "mix_1":
        ch2_name, ch3_name = MIX_1_CH_NAMES
    elif mix == "mix_2":
        ch2_name, ch3_name = MIX_2_CH_NAMES
    else:
        raise ValueError(f"Unknown mix: {mix}")

    # data for whole

    whole_roi_data = {
        "roi": roi,
        "mix": mix,
        "roi_type": apical_type,
        "zone": "whole_roi",
        f"{ch2_name}_total_intensity": ch2_roi.sum(),
        f"{ch3_name}_total_intensity": ch3_roi.sum(),
        f"{ch2_name}_{ch3_name}_correlation": (
            pearsonr(ch2_roi, ch3_roi)[0] if len(ch2_roi) > 0 else 0
        ),
    }

    # data for apical zone

    apical_data = {
        "roi": roi,
        "mix": mix,
        "roi_type": apical_type,
        "zone": "apical",
        f"{ch2_name}_total_intensity": ch2_apical.sum(),
        f"{ch3_name}_total_intensity": ch3_apical.sum(),
        f"{ch2_name}_{ch3_name}_correlation": (
            pearsonr(ch2_apical, ch3_apical)[0] if len(ch2_apical) > 0 else 0
        ),
    }

    if mix == "mix_1":
        mix_1_data.extend([whole_roi_data, apical_data])
    elif mix == "mix_2":
        mix_2_data.extend([whole_roi_data, apical_data])


mix_1_df = pd.DataFrame(mix_1_data)
mix_1_df.to_csv("correlation_mix_1.csv", index=False)


mix_2_df = pd.DataFrame(mix_2_data)
mix_2_df.to_csv("correlation_mix_2.csv", index=False)
