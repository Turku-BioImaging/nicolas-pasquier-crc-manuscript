"""
Plot histograms of inner and outer zones, separated by mix and channel
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

    # data for mix 1 / Cy3 channel / apical-in / outer_zone
    cy3_mix_1_ap_in_outer = []

    for roi in list(root["mix_1"]["apical_in"].keys()):
        cy3 = root["mix_1"]["apical_in"][roi]["raw_data"][:, :, 1]
        outer_mask = root["mix_1"]["apical_in"][roi]["segmentation"]["zones"]["outer"][
            :
        ]
        cy3_masked = cy3[outer_mask]
        cy3_mix_1_ap_in_outer.append(cy3_masked)

    cy3_mix_1_ap_in_outer = np.concatenate(cy3_mix_1_ap_in_outer).flatten()

    # data for mix 1 / Cy3 channel / apical-in / inner_zone
    cy3_mix_1_ap_in_inner = []

    for roi in list(root["mix_1"]["apical_in"].keys()):
        cy3 = root["mix_1"]["apical_in"][roi]["raw_data"][:, :, 1]
        inner_mask = root["mix_1"]["apical_in"][roi]["segmentation"]["zones"]["inner"][
            :
        ]
        cy3_masked = cy3[inner_mask]
        cy3_mix_1_ap_in_inner.append(cy3_masked)

    cy3_mix_1_ap_in_inner = np.concatenate(cy3_mix_1_ap_in_inner).flatten()

    # data for mix 1 / AF647 channel / apical-in / outer_zone
    af647_mix_1_ap_in_outer = []

    for roi in list(root["mix_1"]["apical_in"].keys()):
        af647 = root["mix_1"]["apical_in"][roi]["raw_data"][:, :, 2]
        outer_mask = root["mix_1"]["apical_in"][roi]["segmentation"]["zones"]["outer"][
            :
        ]
        af647_masked = af647[outer_mask]
        af647_mix_1_ap_in_outer.append(af647_masked)

    af647_mix_1_ap_in_outer = np.concatenate(af647_mix_1_ap_in_outer).flatten()

    # data for mix 1 / AF647 channel / apical-in / inner_zone
    af647_mix_1_ap_in_inner = []

    for roi in list(root["mix_1"]["apical_in"].keys()):
        af647 = root["mix_1"]["apical_in"][roi]["raw_data"][:, :, 2]
        inner_mask = root["mix_1"]["apical_in"][roi]["segmentation"]["zones"]["inner"][
            :
        ]
        af647_masked = af647[inner_mask]
        af647_mix_1_ap_in_inner.append(af647_masked)

    af647_mix_1_ap_in_inner = np.concatenate(af647_mix_1_ap_in_inner).flatten()

    # data for mix 2 / Cy3 channel / apical-in / outer_zone
    cy3_mix_2_ap_in_outer = []

    for roi in list(root["mix_2"]["apical_in"].keys()):
        cy3 = root["mix_2"]["apical_in"][roi]["raw_data"][:, :, 1]
        outer_mask = root["mix_2"]["apical_in"][roi]["segmentation"]["zones"]["outer"][
            :
        ]
        cy3_masked = cy3[outer_mask]
        cy3_mix_2_ap_in_outer.append(cy3_masked)

    cy3_mix_2_ap_in_outer = np.concatenate(cy3_mix_2_ap_in_outer).flatten()

    # data for mix 2 / Cy3 channel / apical-in / inner_zone
    cy3_mix_2_ap_in_inner = []
    for roi in list(root["mix_2"]["apical_in"].keys()):
        cy3 = root["mix_2"]["apical_in"][roi]["raw_data"][:, :, 1]
        inner_mask = root["mix_2"]["apical_in"][roi]["segmentation"]["zones"]["inner"][
            :
        ]
        cy3_masked = cy3[inner_mask]
        cy3_mix_2_ap_in_inner.append(cy3_masked)

    cy3_mix_2_ap_in_inner = np.concatenate(cy3_mix_2_ap_in_inner).flatten()

    # data for mix 2 / AF647 channel / apical-in / outer_zone
    af647_mix_2_ap_in_outer = []

    for roi in list(root["mix_2"]["apical_in"].keys()):
        af647 = root["mix_2"]["apical_in"][roi]["raw_data"][:, :, 2]
        outer_mask = root["mix_2"]["apical_in"][roi]["segmentation"]["zones"]["outer"][
            :
        ]
        af647_masked = af647[outer_mask]
        af647_mix_2_ap_in_outer.append(af647_masked)

    af647_mix_2_ap_in_outer = np.concatenate(af647_mix_2_ap_in_outer).flatten()

    # data for mix 2 / AF647 channel / apical-in / inner_zone
    af647_mix_2_ap_in_inner = []

    for roi in list(root["mix_2"]["apical_in"].keys()):
        af647 = root["mix_2"]["apical_in"][roi]["raw_data"][:, :, 2]
        inner_mask = root["mix_2"]["apical_in"][roi]["segmentation"]["zones"]["inner"][
            :
        ]
        af647_masked = af647[inner_mask]
        af647_mix_2_ap_in_inner.append(af647_masked)

    af647_mix_2_ap_in_inner = np.concatenate(af647_mix_2_ap_in_inner).flatten()

    df = pd.DataFrame(
        {
            "Intensity": np.concatenate(
                [
                    cy3_mix_1_ap_in_outer,
                    cy3_mix_1_ap_in_inner,
                    af647_mix_1_ap_in_outer,
                    af647_mix_1_ap_in_inner,
                    cy3_mix_2_ap_in_outer,
                    cy3_mix_2_ap_in_inner,
                    af647_mix_2_ap_in_outer,
                    af647_mix_2_ap_in_inner,
                ]
            ),
            "Data": np.concatenate(
                [
                    np.repeat("Mix 1 Cy3 Outer", len(cy3_mix_1_ap_in_outer)),
                    np.repeat("Mix 1 Cy3 Inner", len(cy3_mix_1_ap_in_inner)),
                    np.repeat("Mix 1 AF647 Outer", len(af647_mix_1_ap_in_outer)),
                    np.repeat("Mix 1 AF647 Inner", len(af647_mix_1_ap_in_inner)),
                    np.repeat("Mix 2 Cy3 Outer", len(cy3_mix_2_ap_in_outer)),
                    np.repeat("Mix 2 Cy3 Inner", len(cy3_mix_2_ap_in_inner)),
                    np.repeat("Mix 2 AF647 Outer", len(af647_mix_2_ap_in_outer)),
                    np.repeat("Mix 2 AF647 Inner", len(af647_mix_2_ap_in_inner)),
                ]
            ),
        }
    )

    sns.set_theme(style="white", palette="muted")
    g = sns.FacetGrid(df, col="Data", height=4, aspect=1, col_wrap=4)

    def hist_with_limit(data, **kwargs):
        plt.hist(data, bins=128, **kwargs)
        plt.xlim([0, 25000])

    # g.map(plt.hist, "Intensity", bins=128)
    g.map(hist_with_limit, "Intensity")

    plt.show()
