import copy
from typing import Tuple, Optional, Union, List

import mmcv
import numpy as np

from mmdet.structures import DetDataSample


class Reverse(object):
    """一个用于反向预处理数据样本的类
        Args: data_samples (:obj:`InstanceData`): Data structure for
                    instance-level annotations or predictions.
    """

    def __init__(self, data_samples: Optional['DetDataSample'], transform_config: List):
        self.data_samples = data_samples
        self.transform_config = transform_config[2:-1]
        self.transform_config.reverse()
        self.scale_factor_list = copy.deepcopy(self.data_samples.scale_factor_list)
        self.img_shape = self.data_samples.img_shape

    def reverse_preprocess(self, bboxes: np.ndarray, segs: np.ndarray = None) -> Union[
        Tuple[np.ndarray, np.ndarray], np.ndarray]:

        """反向预处理边界框和分割图像，返回反向处理后的边界框和分割图像
        Args:
            bboxes (np.ndarray): 预测框 (n,4).
            segs (np.ndarray): 预测分割图 (n,h,w) n为目标数量

        Returns:
            Union[Tuple[ndarray, ndarray], ndarray]: 逆向预处理后的bboxes和segs

        """
        assert bboxes.shape[1] == 4
        bboxes = self._reverse_bbox(bboxes)
        if isinstance(segs, np.ndarray):
            segs = self._reverse_seg(segs)
            return bboxes, segs
        return bboxes

    def _reverse_bbox(self, bboxes: np.ndarray, transforms: List = None):

        """反向处理边界框，包括反向填充、反向翻转、反向裁剪和反向缩放，返回反向处理后的边界框
        """
        if not transforms:
            transforms = self.transform_config
        # for bbox in bboxes:
        for transform in transforms:
            if 'Pad' in transform['type']:
                bboxes = self._reverse_pad(bboxes)
            if 'Flip' in transform['type']:
                bboxes = self._reverse_flip(bboxes)
            if 'Crop' in transform['type']:
                bboxes = self._reverse_crop(bboxes)
            if 'Resize' in transform['type']:
                bboxes = self._reverse_resize(bboxes)
            if 'RandomChoice' == transform['type']:
                bboxes = self._reverse_random_choice(bboxes, transform['transforms'])
        self.scale_factor_list = copy.deepcopy(self.data_samples.scale_factor_list)

        return bboxes

    def _reverse_seg(self, seg_imgs, transforms: List = None):

        """反向处理分割图像，返回反向处理后的分割图像数组
        """
        if not transforms:
            transforms = self.transform_config
        results = []
        for img in seg_imgs:
            for transform in transforms:
                if 'Pad' in transform['type']:
                    img = self._reverse_pad_img(img)
                if 'Flip' in transform['type']:
                    img = self._reverse_flip_img(img)
                if 'Crop' in transform['type']:
                    img = self._reverse_crop_img(img)
                if 'Resize' in transform['type']:
                    img = self._reverse_resize_img(img)
                if 'RandomChoice' == transform['type']:
                    img = self._reverse_random_choice_img(img, transform['transforms'])
            results.append(img)
            self.scale_factor_list = copy.deepcopy(self.data_samples.scale_factor_list)

        return np.array(results)

    def reverse_img(self, img, transforms: List = None):
        if not transforms:
            transforms = self.transform_config
        for transform in transforms:
            if 'Pad' in transform['type']:
                img = self._reverse_pad_img(img)
            if 'Flip' in transform['type']:
                img = self._reverse_flip_img(img)
            if 'Crop' in transform['type']:
                img = self._reverse_crop_img(img)
            if 'Resize' in transform['type']:
                img = self._reverse_resize_img(img)
            if 'RandomChoice' == transform['type']:
                img = self._reverse_random_choice_img(img, transform['transforms'])
        self.scale_factor_list = copy.deepcopy(self.data_samples.scale_factor_list)

        return img

    def _reverse_flip_img(self, img):

        """如果数据样本有翻转属性，则对图像进行水平翻转，返回翻转后的图像
        """
        if self.data_samples.flip:
            img = np.fliplr(img)
        return img

    def _reverse_pad_img(self, img):

        """如果数据样本有填充属性，则对图像进行反向填充，返回去填充后的图像
        """
        # scale_factor = self.scale_factor_list[-1]
        pre_pad_shape = self.data_samples.pre_pad_size
        if img.any():
            img = img[0:pre_pad_shape[0], 0:pre_pad_shape[1], ...]
        return img

    def _reverse_crop_img(self, img):

        """如果数据样本有裁剪属性，则对图像进行反向裁剪，返回还原后的图像
        """
        crop = self.data_samples.crop_index  # x1 x2 y1 y2
        pre_crop_shape = self.data_samples.pre_crop_size
        if crop:
            img = mmcv.impad(img, padding=(crop[0], crop[2], pre_crop_shape[1] - crop[1], pre_crop_shape[0] - crop[3]))
        return img

    def _reverse_resize_img(self, img):

        """对图像进行反向缩放，返回缩放后的图像
        """
        scale_factor = self.scale_factor_list.pop()
        ori_shape = (round(img.shape[0] / scale_factor[0]), round(img.shape[1] / scale_factor[1]))
        img = mmcv.imresize(img, (ori_shape[-1], ori_shape[0]))
        return img

    def _reverse_random_choice_img(self, img, transforms):
        transforms = copy.deepcopy(transforms[self.data_samples.random_choice_idx])
        transforms.reverse()
        img = self.reverse_img(img, transforms)
        return img

    def _reverse_pad(self, bbox):

        """因为填充默认在右、下进行数据填充，不会对边界框造成影响，因此，
        对边界框不做任何填充操作，返回原始的边界框

        """
        self.img_shape = self.data_samples.pre_pad_size
        return bbox

    def _reverse_flip(self, bboxes):

        """如果数据样本有翻转属性，则对边界框进行水平翻转，返回翻转后的边界框
        """
        if self.data_samples.flip:
            if self.data_samples.flip_direction == 'horizontal':
                bboxes[:, 2] = self.img_shape[1] - bboxes[:, 2]
                bboxes[:, 0] = self.img_shape[1] - bboxes[:, 0]
        for bbox in bboxes:
            if bbox[0] > bbox[2]:
                bbox[0], bbox[2] = bbox[2], bbox[0]
            if bbox[1] > bbox[3]:
                bbox[1], bbox[3] = bbox[1], bbox[3]

        return bboxes

    def _reverse_crop(self, bboxes):

        """如果数据样本有裁剪属性，则对边界框进行反向裁剪，返回裁剪后的边界框
        """
        # img_shape hwc
        crop = self.data_samples.crop_index  # x1 x2 y1 y2
        self.img_shape = self.data_samples.pre_crop_size
        if crop:
            bboxes[:, 0] += crop[0]
            bboxes[:, 1] += crop[2]
            bboxes[:, 2] += crop[0]
            bboxes[:, 3] += crop[2]

        return bboxes

    def _reverse_resize(self, bboxes):
        """对边界框进行反向缩放，返回缩放后的边界框
        """
        scale_factor = self.scale_factor_list.pop()
        bboxes[:, 0] /= scale_factor[0]
        bboxes[:, 1] /= scale_factor[1]
        bboxes[:, 2] /= scale_factor[0]
        bboxes[:, 3] /= scale_factor[1]
        self.img_shape = (self.img_shape[0] / scale_factor[0], self.img_shape[1] / scale_factor[1])
        return bboxes

    def _reverse_random_choice(self, bboxes, transforms):
        transforms = copy.deepcopy(transforms[self.data_samples.random_choice_idx])
        transforms.reverse()
        bboxes = self._reverse_bbox(bboxes, transforms)
        return bboxes
