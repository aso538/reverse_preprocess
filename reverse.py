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

    def reverse_preprocess(self, bboxes: np.ndarray, segs: np.ndarray = None) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:

        """反向预处理边界框和分割图像，返回反向处理后的边界框和分割图像
        Args:
            bboxes (np.ndarray): 预测框 (n,4).
            segs (np.ndarray): 预测分割图 (n,h,w) n为目标数量

        Returns:
            Union[Tuple[ndarray, ndarray], ndarray]: 逆向预处理后的bboxes和segs

        """
        assert bboxes.shape[1] == 4
        bboxes = self._reverse_bbox(bboxes)
        # self.data_samples.gt_instances.bboxes = bboxes
        if isinstance(segs, np.ndarray):
            segs = self._reverse_seg(segs)
            # self.data_samples.gt_instances.masks = segs
            return bboxes, segs
        return bboxes

    def _reverse_bbox(self, bboxes: np.ndarray):

        """反向处理边界框，包括反向填充、反向翻转、反向裁剪和反向缩放，返回反向处理后的边界框
        """
        results = []
        for bbox in bboxes:
            for transform in self.transform_config:
                if 'Pad' in transform['type']:
                    bbox = self._reverse_pad(bbox)
                if 'Flip' in transform['type']:
                    bbox = self._reverse_flip(bbox)
                if 'Crop' in transform['type']:
                    bbox = self._reverse_crop(bbox)
                if 'Resize' in transform['type']:
                    bbox = self._reverse_resize(bbox)
            results.append(bbox)

        return np.array(results)

    def _reverse_seg(self, seg_imgs):

        """反向处理分割图像，包括反向填充、反向翻转、反向裁剪和反向缩放，返回反向处理后的分割图像数组
        """
        results = []
        for img in seg_imgs:
            for transform in self.transform_config:
                if 'Pad' in transform['type']:
                    img = self._reverse_pad_img(img)
                if 'Flip' in transform['type']:
                    img = self._reverse_flip_img(img)
                if 'Crop' in transform['type']:
                    img = self._reverse_crop_img(img)
                if 'Resize' in transform['type']:
                    img = self._reverse_resize_img(img)
            results.append(img)

        return np.array(results)

    def reverse_img(self, img):
        for transform in self.transform_config:
            if 'Pad' in transform['type']:
                img = self._reverse_pad_img(img)
            if 'Flip' in transform['type']:
                img = self._reverse_flip_img(img)
            if 'Crop' in transform['type']:
                img = self._reverse_crop_img(img)
            if 'Resize' in transform['type']:
                img = self._reverse_resize_img(img)
        return img

    def _reverse_flip_img(self, img):

        """如果数据样本有翻转属性，则对图像进行水平翻转，返回翻转后的图像
        """
        if self.data_samples.flip:
            img = np.fliplr(img)
        return img

    def _reverse_pad_img(self, img):

        """如果数据样本有填充属性，则对图像进行反向填充，返回填充后的图像
        """
        pre_pad_shape = (
            min(self.data_samples.img_shape[0],
                int(self.data_samples.ori_shape[0] * self.data_samples.scale_factor[0])),
            min(self.data_samples.img_shape[1],
                int(self.data_samples.ori_shape[1] * self.data_samples.scale_factor[1])))
        if img.any():
            img = img[0:pre_pad_shape[0], 0:pre_pad_shape[1], ...]
        return img

    def _reverse_crop_img(self, img):

        """如果数据样本有裁剪属性，则对图像进行反向裁剪，返回裁剪后的图像
        """
        crop = self.data_samples.crop_index  # x1 x2 y1 y2
        pre_crop_shape = (int(self.data_samples.ori_shape[0] * self.data_samples.scale_factor[0]),
                          int(self.data_samples.ori_shape[1] * self.data_samples.scale_factor[1]))
        if crop:
            img = mmcv.impad(img, padding=(crop[0], crop[2], pre_crop_shape[1] - crop[1], pre_crop_shape[0] - crop[3]))
        return img

    def _reverse_resize_img(self, img):

        """对图像进行反向缩放，返回缩放后的图像
        """
        ori_shape = self.data_samples.ori_shape
        img = mmcv.imresize(img, (ori_shape[-1], ori_shape[0]))
        return img

    def _reverse_pad(self, bbox):

        """因为填充默认在右、下进行数据填充，不会对边界框造成影响，因此，
        对边界框不做任何填充操作，返回原始的边界框

        """
        return bbox

    def _reverse_flip(self, bbox):

        """如果数据样本有翻转属性，则对边界框进行水平翻转，返回翻转后的边界框
        """
        if self.data_samples.flip:
            # if self.data_samples.flip_direction == 'horizontal'
            bbox[0], bbox[2] = self.data_samples.img_shape[1] - bbox[2], \
                               self.data_samples.img_shape[1] - bbox[0]

        return bbox

    def _reverse_crop(self, bbox):

        """如果数据样本有裁剪属性，则对边界框进行反向裁剪，返回裁剪后的边界框
        """
        # img_shape hwc
        crop = self.data_samples.crop_index  # x1 x2 y1 y2
        if crop:
            bbox[0] = bbox[0] + crop[0]
            bbox[1] = bbox[1] + crop[2]
            bbox[2] = bbox[2] + crop[0]
            bbox[3] = bbox[3] + crop[2]

        return bbox

    def _reverse_resize(self, bbox):
        """对边界框进行反向缩放，返回缩放后的边界框
        """
        scale_factor = self.data_samples.scale_factor
        bbox[0] /= scale_factor[0]
        bbox[1] /= scale_factor[1]
        bbox[2] /= scale_factor[0]
        bbox[3] /= scale_factor[1]
        return bbox
