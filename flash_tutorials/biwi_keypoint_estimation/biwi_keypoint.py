#!/usr/bin/env python

import os
from pathlib import Path

import flash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

datadir = Path("/kaggle/input")


from flash.core.utilities.imports import _ICEVISION_AVAILABLE
from flash.image.data import IMG_EXTENSIONS, NP_EXTENSIONS, image_loader

if _ICEVISION_AVAILABLE:
    from icevision.core.record import BaseRecord
    from icevision.core.record_components import ClassMapRecordComponent, FilepathRecordComponent, tasks
    from icevision.data.data_splitter import SingleSplitSplitter
    from icevision.parsers.parser import Parser
else:
    assert 0, "with ice please"


# In[108]:


from pathlib import Path

from icevision.all import *
from icevision.core.record_defaults import KeypointsRecord


class CustomBIWIKeypointsMetadata(KeypointsMetadata):
    labels = ["center"]  # , "apex", "root"]


class BiwiPNG:
    # TODO cache calibration for each subject (avoid loading for every frame)

    def load_keypoints(self, impath):
        name = str(impath)[:-8]

        pose = np.loadtxt(name + "_pose.txt")
        R = pose[:3, :3]  # pose rotation from standard pose to this
        centre_biwi = pose[3, :]

        cal_rgb = os.path.join(os.path.split(name)[0], "rgb.cal")
        cal_rgb_P = np.eye(4)
        cal_rgb_P[:3, :3] = np.genfromtxt(cal_rgb, skip_header=5, skip_footer=2)
        cal_rgb_P[:3, 3] = np.genfromtxt(cal_rgb, skip_header=9, skip_footer=1)
        cal_rgb = np.genfromtxt(cal_rgb, skip_footer=6)

        def biwi2img(vec, camera_cal=True):
            if camera_cal:  # RGB camera calibration
                x, y, z = cal_rgb_P[:3, :3] @ vec + cal_rgb_P[:3, 3]
            else:
                x, y, z = vec
            # BIWI world to image conversion
            # x <--> v
            # y <--> u
            # z  ==  d
            v = x * cal_rgb[0, 0] / z + cal_rgb[0, 2]
            u = y * cal_rgb[1, 1] / z + cal_rgb[1, 2]
            return u, v

        centre = biwi2img(centre_biwi)
        # assuming the standard orientation of the nose is frontal upright, apex and root distance and directions are guesses
        dist = 50.0
        apex = biwi2img(centre_biwi + dist * R @ np.array([0, 0, -1.0]))
        root = biwi2img(
            centre_biwi + dist / np.sqrt(2) * R @ np.array([0, -1.0, -1.0])
        )  # guessed 45 degree angle towards root

        return {"center": centre, "apex": apex, "root": root}


class CustomParser(Parser):
    def __init__(self, img_dir: Union[str, Path], imgID_annotations: Dict, idmap=None):
        super().__init__(template_record=self.template_record(), idmap=idmap)

        self.img_dir = Path(img_dir)

        self.class_map = ClassMap(CustomBIWIKeypointsMetadata().labels)

        self.annotations_dict = imgID_annotations

    def __iter__(self):
        yield from self.annotations_dict.items()

    def __len__(self):
        return len(self.annotations_dict)

    def template_record(self) -> BaseRecord:
        return KeypointsRecord()

    def record_id(self, o):
        return o[0]

    def filepath(self, o):
        return self.img_dir / o[0]

    def keypoints(self, o):
        return [
            KeyPoints.from_xyv([x, y, 1], CustomBIWIKeypointsMetadata) for y, x in o[1]
        ]  # TODO check coordinate flip

    def image_width_height(self, o) -> Tuple[int, int]:
        return get_img_size(self.filepath(o))

    def labels(self, o) -> List[Hashable]:
        return list(range(1, len(CustomBIWIKeypointsMetadata().labels) + 1))

    def bboxes(self, o) -> List[BBox]:
        w, h = get_img_size(self.filepath(o))
        return [BBox.from_xywh(0, 0, w, h)] * (len(CustomBIWIKeypointsMetadata().labels))

    def parse_fields(self, o, record, is_new):
        if is_new:
            record.set_filepath(self.filepath(o))
            record.set_img_size(self.image_width_height(o))

        record.detection.set_class_map(self.class_map)
        record.detection.add_labels_by_id(self.labels(o))
        record.detection.add_bboxes(self.bboxes(o))
        record.detection.add_keypoints(self.keypoints(o))


def parser(data_dir: Path):

    images = sorted(Path(data_dir).glob("??/frame_*_rgb.png"))[:100]  # TODO remove truncation

    imgID_annotations = {}

    biwi = BiwiPNG()
    for im in images:
        keypoints = biwi.load_keypoints(im)
        imgID_annotations[str(im.relative_to(data_dir))] = [keypoints["center"]]  # TODO add other keypoints

    return CustomParser(img_dir=data_dir, imgID_annotations=imgID_annotations)


if True:
    p = parser(datadir)
    p.parse()

    for s in p:
        break

    r = KeypointsRecord()
    p.parse_fields(s, r, True)

    for kp in p.keypoints(s):
        print(kp.xyv)

    print(s, r)


# In[109]:


from flash.image import KeypointDetectionData, KeypointDetector

datamodule = KeypointDetectionData.from_icedata(parser=parser, train_folder=datadir, batch_size=8)

model = KeypointDetector(
    head="keypoint_rcnn",
    backbone="resnet18_fpn",
    num_keypoints=3,
    num_classes=3,
)

trainer = flash.Trainer(max_epochs=2, gpus=1)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")


# In[110]:


sample = datamodule.train_dataset[0]
sample


# In[111]:


from flash.core.data.io.input import DataKeys

plt.imshow(sample[DataKeys.INPUT])
plt.scatter(
    sample[DataKeys.TARGET]["keypoints"][0][0]["x"], sample[DataKeys.TARGET]["keypoints"][0][0]["y"], marker="+"
)
sample


# In[ ]:


# In[ ]:


# In[ ]:
