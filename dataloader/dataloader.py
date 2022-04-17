from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os

from utils import utils
from utils.file_io import read_img, read_disp

import json ## add to act with bounding boxes


class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform

        sceneflow_finalpass_dict = {
            'train': 'filenames/SceneFlow_finalpass_train.txt',
            'val': 'filenames/SceneFlow_finalpass_val.txt',
            'test': 'filenames/SceneFlow_finalpass_test.txt'
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            # 'test': 'filenames/KITTI_2012_test.txt'
            'test': 'filenames/KITTI_2012_val_one.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': 'filenames/KITTI_mix.txt',
            'test': 'filenames/KITTI_2015_test.txt'
        }

        ## Add apolloscape dataset
        apolloscape_dict = {
            'train': 'filenames/apolloscape_train.txt',
            'val': 'filenames/apolloscape_val.txt',
            # 'test': 'filenames/apolloscape_test.txt'
            'test': 'filenames/apolloscape_one_img.txt'
        }
        ## End adding

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
            'apolloscape': apolloscape_dict, ## Add apolloscape dataset to dataset dictionary
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        left_json_file = 'filenames/stereo_test_left_traffic_result.json'
        left_bbox_file = open(left_json_file)
        left_bboxes = json.load(left_bbox_file)

        right_json_file = 'filenames/stereo_test_right_traffic_result.json'
        right_bbox_file = open(right_json_file)
        right_bboxes = json.load(right_bbox_file)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]

            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None

            if load_pseudo_gt and sample['disp'] is not None:
                # KITTI 2015
                if 'disp_occ_0' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ_0',
                                                                     'disp_occ_0_pseudo_gt')
                # KITTI 2012
                elif 'disp_occ' in sample['disp']:
                    sample['pseudo_disp'] = (sample['disp']).replace('disp_occ',
                                                                     'disp_occ_pseudo_gt')
                else:
                    raise NotImplementedError
            else:
                sample['pseudo_disp'] = None

            ### Add attribute 'left_bbox' to the sample
            img_name = left_img[-29:-13]
            imgL = [img for img in left_bboxes if img_name in img['filename']]
            imgR = [img for img in right_bboxes if img_name in img['filename']]

            imgL = left_bboxes[0]
            imgR = right_bboxes[0]

            sample['left_bbox'] = imgL['objects']
            sample['right_bbox'] = imgR['objects']

            self.samples.append(sample)


    def bboxes(self, objects, img_width, img_height):
        bbox_list = []
        for obj in objects:
            c = obj['class_id']
            center_x = obj['relative_coordinates']['center_x']
            center_y = obj['relative_coordinates']['center_y']
            w_obj = obj['relative_coordinates']['width']
            h_obj = obj['relative_coordinates']['height']

            x_center = int(float(center_x)*img_width)
            y_center = int((float(center_y))*img_height)
            w = int(float(w_obj)*img_width)
            h = int(float(h_obj)*img_height)

            # PIL coordinates are different from OpenCV coordinates
            x_min = int(x_center - w/2)
            y_min = int(y_center - h/2)

            bbox = [c, x_min, y_min, x_min + w, y_min + h]

            bbox_list.append(bbox)
        return bbox_list
        pass        

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        img_height = sample['left'].shape[0]
        img_width = sample['left'].shape[1]

        ### Bboxes
        sample['left_bboxes'] = self.bboxes(sample_path['left_bbox'], img_width, img_height)
        sample['right_bboxes'] = self.bboxes(sample_path['right_bbox'], img_width, img_height) 
        # import pdb; pdb.set_trace()

        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = read_disp(sample_path['pseudo_disp'], subset=subset)  # [H, W]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
