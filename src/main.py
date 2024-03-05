"""
Main entrypoint script for ROI segmentation and analysis.
"""

import argparse

import zarr
from joblib import Parallel, delayed
from segmentation_classes import ApicalInSegmenter, ApicalOutSegmenter
from zoning_classes import ApicalInZoner, ApicalOutZoner


def segment_roi(zarr_path: str, mix: str, roi: str):
    if "_in_" in roi:
        segmenter = ApicalInSegmenter(zarr_path, mix, roi)
        segmenter.segment()
    elif "_out_" in roi:
        segmenter = ApicalOutSegmenter(zarr_path, mix, roi)
        segmenter.segment()
    else:
        raise ValueError(f"Cannot infer apical type of ROI {roi}")


def zone_roi(zarr_path: str, mix: str, roi: str):
    if "_in_" in roi:
        zoner = ApicalInZoner(zarr_path, mix, roi)
        zoner.generate()
    elif "_out_" in roi:
        zoner = ApicalOutZoner(zarr_path, mix, roi)
        zoner.generate()
    else:
        raise ValueError(f"Cannot infer apical type of ROI {roi}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment apical-in and apical-out ROIs. Outputs are saved to Zarr datasets."
    )

    parser.add_argument(
        "--zarr-path", type=str, required=True, help="Path to the Zarr file."
    )

    parser.add_argument(
        "--skip-roi-segmentation",
        action="store_true",
        help="Skip the ROI segmentation step",
    )

    parser.add_argument(
        "--skip-zoning", action="store_true", help="Skip the zoning step"
    )

    args = parser.parse_args()

    root = zarr.open(args.zarr_path, mode="r")

    if args.skip_roi_segmentation is False:
        mixes = list(root.keys())
        assert len(mixes) == 2, "Expected 2 mixes in the Zarr file"

        rois = []
        for mix in mixes:
            rois.extend(
                (args.zarr_path, mix, roi)
                for roi in list(root[mix]["apical_in"].keys())
                + list(root[mix]["apical_out"].keys())
            )

        print("Segmenting ROIs...")
        Parallel(n_jobs=-1, verbose=10)(delayed(segment_roi)(*roi) for roi in rois)

    if args.skip_zoning is False:
        mixes = list(root.keys())
        assert len(mixes) == 2, "Expected 2 mixes in the Zarr file"

        rois = []
        for mix in mixes:
            rois.extend(
                (args.zarr_path, mix, roi)
                for roi in list(root[mix]["apical_in"].keys())
                + list(root[mix]["apical_out"].keys())
            )

        print("Generating ROI zones...")
        Parallel(n_jobs=6, verbose=10)(delayed(zone_roi)(*roi) for roi in rois)
