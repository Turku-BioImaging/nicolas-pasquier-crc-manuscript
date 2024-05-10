"""
Contains classes for nuclei segmentation
"""

import numpy as np
import zarr
from skimage import img_as_uint
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from skimage.measure import regionprops

SIZE_THRESHOLD = 150


class NucleiSegmenter:

    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    def __init__(self, zarr_path: str, sample_path: str):
        self.zarr_path = zarr_path
        self.root = zarr.open(zarr_path, mode="a")
        self.sample_path = sample_path

    def segment(self) -> np.ndarray:
        """
        Segments the nuclei in the sample image and stores the segmentation labels in a dataset.

        This method applies a pre-trained StarDist model to predict the instances of nuclei in the sample image.
        It then removes the labels of regions that overlap with the background or are smaller than a size threshold.
        The segmentation labels are stored in a Zarr dataset.

        Returns:
        np.ndarray: The segmentation labels of the nuclei in the sample image.
        """
        dapi = self.root[self.sample_path]["raw_data"][:, :, 0]

        mask = np.array(self.root[self.sample_path]["segmentation"]["mask"])

        dapi[~mask] = 0

        labels, _ = self.model.predict_instances(normalize(dapi))

        mask_bg = mask == 0
        for item in regionprops(labels):
            overlap = np.logical_and((labels == item.label), mask_bg)
            if np.any(overlap) or item.area < SIZE_THRESHOLD:
                labels[labels == item.label] = 0

        labels = img_as_uint(labels)

        # write zarr dataset
        label_path = f"{self.sample_path}/segmentation/nuclei"

        if label_path in self.root:
            del self.root[label_path]

        label_dataset = self.root.create_dataset(label_path, data=labels)
        label_dataset.attrs.update(
            {"author": "Turku BioImaging", "description": "Nuclei segmentation labels"}
        )

        return labels
