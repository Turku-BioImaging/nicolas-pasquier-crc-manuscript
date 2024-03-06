"""
Plot histograms of the data, separated by mix, apical type, and channel
"""

import pandas as pd
import numpy as np
import os
import zarr

import seaborn as sns
import matplotlib.pyplot as plt

ZARR_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "zarr_data", "roi_data.zarr"
)

if __name__ == "__main__":
    root = zarr.open(ZARR_PATH, mode="r")

    # gather for Cy3 channel / mix 1 / apical-in
    cy3_mix_1_ap_in = []

    for roi in list(root["mix_1"]["apical_in"].keys()):
        cy3 = root["mix_1"]["apical_in"][roi]["raw_data"][:, :, 1]
        mask = root["mix_1"]["apical_in"][roi]["segmentation"]["mask"][:]
        cy3_masked = cy3[mask]
        cy3_mix_1_ap_in.append(cy3_masked)

    cy3_mix_1_ap_in = np.concatenate(cy3_mix_1_ap_in).flatten()

    # gather for Cy3 channel / mix 1 / apical-out
    cy3_mix_1_ap_out = []

    for roi in list(root["mix_1"]["apical_out"].keys()):
        cy3 = root["mix_1"]["apical_out"][roi]["raw_data"][:, :, 1]
        mask = root["mix_1"]["apical_out"][roi]["segmentation"]["mask"][:]
        cy3_masked = cy3[mask]
        cy3_mix_1_ap_out.append(cy3_masked)

    cy3_mix_1_ap_out = np.concatenate(cy3_mix_1_ap_out).flatten()

    # gather data for AF647 channel / mix 1 / apical-in
    af647_mix_1_ap_in = []

    for roi in list(root["mix_1"]["apical_in"].keys()):
        af647 = root["mix_1"]["apical_in"][roi]["raw_data"][:, :, 2]
        mask = root["mix_1"]["apical_in"][roi]["segmentation"]["mask"][:]
        af647_masked = af647[mask]
        af647_mix_1_ap_in.append(af647_masked)

    af647_mix_1_ap_in = np.concatenate(af647_mix_1_ap_in).flatten()

    # gather data for AF647 channel / mix 1 / apical-out
    af647_mix_1_ap_out = []

    for roi in list(root["mix_1"]["apical_out"].keys()):
        af647 = root["mix_1"]["apical_out"][roi]["raw_data"][:, :, 2]
        mask = root["mix_1"]["apical_out"][roi]["segmentation"]["mask"][:]
        af647_masked = af647[mask]
        af647_mix_1_ap_out.append(af647_masked)

    af647_mix_1_ap_out = np.concatenate(af647_mix_1_ap_out).flatten()

    # gather data for Cy3 channel / mix 2 / apical-in
    cy3_mix_2_ap_in = []

    for roi in list(root["mix_2"]["apical_in"].keys()):
        cy3 = root["mix_2"]["apical_in"][roi]["raw_data"][:, :, 1]
        mask = root["mix_2"]["apical_in"][roi]["segmentation"]["mask"][:]
        cy3_masked = cy3[mask]
        cy3_mix_2_ap_in.append(cy3_masked)

    cy3_mix_2_ap_in = np.concatenate(cy3_mix_2_ap_in).flatten()

    # gather data for Cy3 channel / mix 2 / apical-out
    cy3_mix_2_ap_out = []

    for roi in list(root["mix_2"]["apical_out"].keys()):
        cy3 = root["mix_2"]["apical_out"][roi]["raw_data"][:, :, 1]
        mask = root["mix_2"]["apical_out"][roi]["segmentation"]["mask"][:]
        cy3_masked = cy3[mask]
        cy3_mix_2_ap_out.append(cy3_masked)

    cy3_mix_2_ap_out = np.concatenate(cy3_mix_2_ap_out).flatten()

    # gather data for AF647 channel / mix 2 / apical-in
    af647_mix_2_ap_in = []

    for roi in list(root["mix_2"]["apical_in"].keys()):
        af647 = root["mix_2"]["apical_in"][roi]["raw_data"][:, :, 2]
        mask = root["mix_2"]["apical_in"][roi]["segmentation"]["mask"][:]
        af647_masked = af647[mask]
        af647_mix_2_ap_in.append(af647_masked)

    af647_mix_2_ap_in = np.concatenate(af647_mix_2_ap_in).flatten()

    # gather data for AF647 channel / mix 2 / apical-out
    af647_mix_2_ap_out = []

    for roi in list(root["mix_2"]["apical_out"].keys()):
        af647 = root["mix_2"]["apical_out"][roi]["raw_data"][:, :, 2]
        mask = root["mix_2"]["apical_out"][roi]["segmentation"]["mask"][:]
        af647_masked = af647[mask]
        af647_mix_2_ap_out.append(af647_masked)

    af647_mix_2_ap_out = np.concatenate(af647_mix_2_ap_out).flatten()

    df = pd.DataFrame(
        {
            "Intensity": np.concatenate(
                [
                    cy3_mix_1_ap_in,
                    cy3_mix_1_ap_out,
                    af647_mix_1_ap_in,
                    af647_mix_1_ap_out,
                    cy3_mix_2_ap_in,
                    cy3_mix_2_ap_out,
                    af647_mix_2_ap_in,
                    af647_mix_2_ap_out,
                ]
            ),
            "Data": np.concatenate(
                [
                    np.repeat("Mix 1 Cy3 ap_in", len(cy3_mix_1_ap_in)),
                    np.repeat("Mix 1 Cy3 ap_out", len(cy3_mix_1_ap_out)),
                    np.repeat("Mix 1 AF647 ap_in", len(af647_mix_1_ap_in)),
                    np.repeat("Mix 1 AF647 ap_out", len(af647_mix_1_ap_out)),
                    np.repeat("Mix 2 Cy3 ap_in", len(cy3_mix_2_ap_in)),
                    np.repeat("Mix 2 Cy3 ap_out", len(cy3_mix_2_ap_out)),
                    np.repeat("Mix 2 AF647 ap_in", len(af647_mix_2_ap_in)),
                    np.repeat("Mix 2 AF647 ap_out", len(af647_mix_2_ap_out)),
                ]
            ),
        }
    )

    # sns.color_palette('hls', 8)
    sns.set_theme(style="white", palette="muted")

    g = sns.FacetGrid(df, col="Data", height=4, aspect=1, col_wrap=4)

    def hist_with_limit(data, **kwargs):
        plt.hist(data, bins=128, **kwargs)
        plt.xlim([0, 25000])

    # g.map(plt.hist, "Intensity", bins=128)
    g.map(hist_with_limit, "Intensity")
    plt.show()
