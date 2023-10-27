import mmcv
import json

from mmcv import Compose

from mmdet.registry import TRANSFORMS
from mmdet.utils import register_all_modules
from mmdet.visualization.local_visualizer import DetLocalVisualizer

from reverse import Reverse

register_all_modules()

trm_det_train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(
    #     type='CachedMosaic',
    #     img_scale=(640, 640),
    #     pad_val=114.0,
    #     max_cached_images=20,
    #     random_pop=False),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=1.0),
    dict(type='Pad', size=(800, 800), pad_val=dict(img=(114, 114, 114))),
    dict(type='PackDetInputs',
         meta_keys=(
             'img_id', 'img_path', 'ori_shape', 'img_shape',
             'scale_factor', 'flip', 'flip_direction', 'crop_index', 'scale_factor_list',
             'pre_pad_size', 'pre_crop_size'
         )
         )
]

dino_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=1.0),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    # scales=(480,1333),
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction', 'crop_index', 'scale_factor_list',
                    'random_choice_idx', 'pre_pad_size', 'pre_crop_size'
                    )
         )
]

mask_rcnn_train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=1.0),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction', 'crop_index', 'scale_factor_list',
                    'pre_pad_size', 'pre_crop_size'
                    )
         )
]


def tensor2numpy(tensor):
    """
    将tensor格式数据转换为图片
    :param tensor:

    :return:
    """
    return tensor.cpu().detach().numpy().transpose(1, 2, 0)


def build_data_preprocess(transform_config):
    transforms = []
    for transform in transform_config:
        if isinstance(transform, dict):
            transforms.append(TRANSFORMS.build(transform))
        else:
            transforms.append(transform)
    pipeline = Compose(transforms)

    return pipeline


def get_labels(img_path=None):
    with open(r'data/balloon_dataset/annotations/trainval.json', "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not img_path:
        img_path = r'data\balloon\train\154446334_5d41cd1375_b.jpg'
    all_label = {}
    for i in labels['annotations']:
        img_id = i['image_id']
        if labels['images'][img_id]['file_name'] in all_label.keys():
            all_label[labels['images'][img_id]['file_name']]['labels'].append(i)
        else:
            all_label[labels['images'][img_id]['file_name']] = {'image': labels['images'][img_id], 'labels': [i]}
    if not isinstance(img_path, (list, tuple)):
        img_path = [img_path]
    data_infos = []
    for i in img_path:
        i = i.replace('\\', '/')
        file_name = i.split('/')[-1]
        data = all_label[file_name]
        img_label = []
        for label in data['labels']:
            img_label.append({
                'ignore_flag': 0,
                'bbox': [
                    label['bbox'][0],
                    label['bbox'][1],
                    label['bbox'][0] + label['bbox'][2],
                    label['bbox'][1] + label['bbox'][3]
                ],
                'bbox_label': label['category_id'] - 1,
                'mask': [label['segmentation']]
            })
        data_infos.append({
            'img_path': i,
            'img_id': data['image']['id'],
            'seg_map_path': None,
            'height': data['image']['height'],
            'width': data['image']['width'],
            'instances': img_label,
            'sample_idx': data['image']['id']
        })
    return data_infos


def main():
    save_path = './out'
    img_paths = [
        r'data\balloon\train\154446334_5d41cd1375_b.jpg',
        r'data\balloon\train\7488015492_0583857ca0_k.jpg',
    ]
    transform_config = mask_rcnn_train_pipeline
    data_infos = get_labels(img_paths)
    vis = DetLocalVisualizer()
    pipeline = build_data_preprocess(transform_config=transform_config)
    print(pipeline)

    for data_info in data_infos:
        pre_result = pipeline(data_info)
        data_samples = pre_result['data_samples']
        reverse = Reverse(data_samples, transform_config)
        print(data_samples)

        bboxes = data_samples.gt_instances.bboxes.tensor.numpy()
        data_samples.gt_instances.bboxes = bboxes
        segs = data_samples.gt_instances.masks.masks
        img_path = data_info['img_path']
        img_name = img_path.split('/')[-1]
        # 683,1024-> 548 824 -> 548 640 -> 640 640
        # ori_size-> resize -> crop -> pad

        img = pre_result['inputs']
        img = tensor2numpy(img)
        img = mmcv.rgb2bgr(img)
        vis.add_datasample(name='', image=img, data_sample=data_samples, draw_pred=False, show=False,
                           out_file='./{}/{}_数据增强.jpg'.format(save_path, img_name.split('.')[0]))

        bboxes, segs = reverse.reverse_preprocess(bboxes, segs)
        data_samples.gt_instances.bboxes = bboxes
        data_samples.gt_instances.masks.masks = segs

        img = reverse.reverse_img(img)

        vis.add_datasample(name='', image=img, data_sample=data_samples, draw_pred=False, show=False,
                           out_file='./{}/{}_还原.jpg'.format(save_path, img_name.split('.')[0]))

        image = mmcv.rgb2bgr(mmcv.imread(img_path))
        image = mmcv.imresize(image, size=(img.shape[1], img.shape[0]))
        vis.add_datasample(name='', image=image, data_sample=data_samples,
                           draw_pred=False, show=False,
                           out_file='./{}/{}_还原_原图.jpg'.format(save_path, img_name.split('.')[0]))


if __name__ == '__main__':
    main()
    # get_labels()
