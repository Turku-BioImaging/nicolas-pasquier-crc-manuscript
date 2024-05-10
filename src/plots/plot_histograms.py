"""
Plot intensity histograms of the different mixes, zones, and channels.
"""

import os
import zarr
import numpy as np
import matplotlib.pyplot as plt

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "zarr_data", "roi_data.zarr"
)

root = zarr.open(ZARR_PATH, mode="r")

mixes = list(root.keys())
roi_types = list(root[mixes[0]].keys())

# data for mix 1 / apical_in / ch 2
mix_1_apical_in_whole_roi_ch2 = []
mix_1_apical_in_whole_roi_ch3 = []

for roi in list(root['mix_1']['apical_in'].keys()):
    print(roi)