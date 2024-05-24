import numpy as np
import os
from nuclei_segmentation import NucleiSegmenter
import zarr
from skimage import io, img_as_uint
from skimage.measure import regionprops
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import pandas as pd

ZARR_PATH = os.path.join("..", "zarr_data", "roi_data.zarr")


root = zarr.open(ZARR_PATH, mode="r")
# mixes = list(root.keys())

dapi = root["mix_1"]["apical_in"]["231019_mix1_8_in_1"]["raw_data"][:, :, 0]
io.imshow(dapi)

mask = np.array(
    root["mix_1"]["apical_in"]["231019_mix1_8_in_1"]["segmentation"]["mask"]
)

dapi[~mask] = 0


model = StarDist2D.from_pretrained("2D_versatile_fluo")

labels, _ = model.predict_instances(normalize(dapi))


sizes = []

mask_bg = mask == 0
for item in regionprops(labels):
    sizes.append(item.area)

    overlap = np.logical_and((labels == item.label), mask_bg)
    if np.any(overlap) or item.area < 100:
        labels[labels == item.label] = 0


io.imshow(labels)
io.imsave("labels.tif", img_as_uint(labels), check_contrast=False)

size_df = pd.DataFrame(sizes, columns=["size"])
size_df.describe()
