"""Peter Dataset Classes

Based on VOC Dataset Classes
(Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot)
"""
from .config import HOME
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import json
import pandas as pd
import numpy as np

PETER_CLASSES = ("text")

PETER_ROOT = osp.join(HOME, "data/peter_ds/")


class PeterAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: just one class)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, anno_df, class_to_ind=None):
        self.class_to_ind = {"text": 1}
        self.anno_df = anno_df

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        bboxes = self.anno_df[self.anno_df["image_id"] == target]["bbox"].to_list()
        for bbox in bboxes:

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox[i]) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height #TODO check if we need it
                bndbox.append(cur_pt)
            label_idx = 1
            bndbox.append(label_idx)
            res += [bndbox]
        return res


class PeterDetection(data.Dataset):
    """Peter Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root,
                 transform=None,
                 dataset_name='Peter',
                 mode = "train"):
        self.root = root
        self.image_df, self.anno_df = self.make_tables(osp.join(self.root, "annotations_" + mode + ".json"))
        self.transform = transform
        self.target_transform = PeterAnnotationTransform(self.annno_df)
        self.name = dataset_name
        self.ids = self.image_df["id"].to_list()

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)
    
    def make_tables(self, json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        f.close()

        image_df = pd.DataFrame(data["images"])
        anno_df = pd.DataFrame(data["annotations"])

        return image_df, anno_df

    def pull_item(self, index):
        img_id = self.ids[index]

        target = self.pull_anno(index)
        img = self.pull_image(index)
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(img_id, width, height)

        if self.transform is not None:
            if self.target_transform is None:
                target = self.target_transform(img_id, width, height)
            target = np.array(self.pull_anno(index))
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(osp.join(PETER_ROOT, "images", self.image_df[self.image_df["id"] == img_id]["file_name"].to_string(index=False)), cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        gt = self.target_transform(img_id, 1, 1)
        return img_id, gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
