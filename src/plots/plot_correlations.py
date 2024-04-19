"""
Plots Pearson correlation distributions for mixes 1 and 2.
Mix 1: HER2 Ch2 / SORLA Ch3
Mix 2: HER2 Ch2 / HER3 Ch3
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "analysis")

# plot data for Mix 1
mix_1_df = pd.read_csv(os.path.join(DATA_DIR, "correlation_mix_1.csv"))
# print(mix_1_df.head())

# plot the following
# - apical-in / whole_roi / corr
# - apical-in / basal / corr
# - apical-in / apical / corr
# - apical-out / apical / corr

apical_in_whole_roi = mix_1_df[
    (mix_1_df["roi_type"] == "apical_in") & (mix_1_df["zone"] == "whole_roi")
]

apical_in_basal = mix_1_df[
    (mix_1_df["roi_type"] == "apical_in") & (mix_1_df["zone"] == "basal")
]

apical_in_apical = mix_1_df[
    (mix_1_df["roi_type"] == "apical_in") & (mix_1_df["zone"] == "apical")
]

df = pd.concat(
    [
        apical_in_whole_roi.assign(zone="Whole ROI"),
        apical_in_basal.assign(zone="Basal zone"),
        apical_in_apical.assign(zone="Apical zone"),
    ]
)

# Create a boxplot
sns.color_palette("viridis")
sns.set_style("white")
plt.figure(figsize=(5, 4))
sns.boxplot(x="zone", y="HER2_SORLA_correlation", data=df, hue="zone")

# Show the plot
plt.show()

# apical out whole roi
apical_out_whole_roi = mix_1_df[
    (mix_1_df["roi_type"] == "apical_out") & (mix_1_df["zone"] == "whole_roi")
]

apical_out_apical = mix_1_df[
    (mix_1_df["roi_type"] == "apical_out") & (mix_1_df["zone"] == "apical")
]

df = pd.concat(
    [
        apical_out_whole_roi.assign(zone="Whole ROI"),
        apical_out_apical.assign(zone="Apical zone"),
    ]
)

sns.color_palette("viridis")
sns.set_style("white")
plt.figure(figsize=(5, 4))
sns.boxplot(x="zone", y="HER2_SORLA_correlation", data=df, hue="zone")
plt.show()


# plot data for Mix 2
mix_2_df = pd.read_csv(os.path.join(DATA_DIR, "correlation_mix_2.csv"))

apical_in_whole_roi = mix_2_df[
    (mix_2_df["roi_type"] == "apical_in") & (mix_2_df["zone"] == "whole_roi")
]

apical_in_basal = mix_2_df[
    (mix_2_df["roi_type"] == "apical_in") & (mix_2_df["zone"] == "basal")
]

apical_in_apical = mix_2_df[
    (mix_2_df["roi_type"] == "apical_in") & (mix_2_df["zone"] == "apical")
]

df = pd.concat(
    [
        apical_in_whole_roi.assign(zone="Whole ROI"),
        apical_in_basal.assign(zone="Basal zone"),
        apical_in_apical.assign(zone="Apical zone"),
    ]
)

# Create a boxplot
sns.color_palette("viridis")
sns.set_style("white")
plt.figure(figsize=(5, 4))
sns.boxplot(x="zone", y="HER2_HER3_correlation", data=df, hue="zone")

# Show the plot
plt.show()

apical_out_whole_roi = mix_2_df[
    (mix_2_df["roi_type"] == "apical_out") & (mix_2_df["zone"] == "whole_roi")
]
apical_out_apical = mix_2_df[
    (mix_2_df["roi_type"] == "apical_out") & (mix_2_df["zone"] == "apical")
]

df = pd.concat(
    [
        apical_out_whole_roi.assign(zone="Whole ROI"),
        apical_out_apical.assign(zone="Apical zone"),
    ]
)

sns.color_palette("viridis")
sns.set_style("white")
plt.figure(figsize=(5, 4))
sns.boxplot(x="zone", y="HER2_HER3_correlation", data=df, hue="zone")
plt.show()
