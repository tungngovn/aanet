from fileinput import filename
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import skimage.io
import argparse
import numpy as np
import time
import os

import nets
import dataloader
from dataloader import transforms
from utils import utils
from utils.file_io import write_pfm

## Tung added libraries
import pdb
from PIL import Image, ImageDraw
from metric import d1_metric, thres_metric, dist_err
import csv
import wandb
wandb.login()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')

# Training data
parser.add_argument('--data_dir', default='data/SceneFlow',
                    type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=1, type=int, help='Batch size for inference')
parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=576, type=int, help='Image height for inference')
parser.add_argument('--img_width', default=960, type=int, help='Image width for inference')

# Model
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--output_dir', default='output', type=str,
                    help='Directory to save inference results')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')

# AANet
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')

parser.add_argument('--save_type', default='png', choices=['pfm', 'png', 'npy'], help='Save file type')
parser.add_argument('--visualize', action='store_true', help='Visualize disparity map')

# Log
parser.add_argument('--count_time', action='store_true', help='Inference on a subset for time counting only')
parser.add_argument('--num_images', default=100, type=int, help='Number of images for inference')

args = parser.parse_args()

model_name = os.path.basename(args.pretrained_aanet)[:-4]
model_dir = os.path.basename(os.path.dirname(args.pretrained_aanet))
args.output_dir = os.path.join(args.output_dir, model_dir + '-' + model_name)

utils.check_path(args.output_dir)
utils.save_command(args.output_dir)

wandb.init(project='AANet+', entity='nttung1cmc', config=args)


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test loader
    test_transform = transforms.Compose([
        transforms.RandomCrop(args.img_height, args.img_width, validate=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    test_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                         dataset_name=args.dataset_name,
                                         mode=args.mode,
                                         save_filename=True,
                                         transform=test_transform)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True, drop_last=False)

    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)

    if os.path.exists(args.pretrained_aanet):
        print('=> Loading pretrained AANet:', args.pretrained_aanet)
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
    else:
        print('=> Using random initialization')

    # Save parameters
    num_params = utils.count_parameters(aanet)
    print('=> Number of trainable parameters: %d' % num_params)

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    ##[=======>>>Depth<<<==================]##
    ### Prepare to calculate depth
    from Disp2Depth import Disp2Depth

    ## Camera parameters
    K = np.array([[2301.3147, 0, 1489.8536], [0, 2301.3147, 479.1750],[0, 0, 1]]) # intrinsic matrix of apolloscape dataset
    baseline = 0.622

    ## Init disp to depth class
    Sys = Disp2Depth(K, baseline)    

    # Inference
    aanet.eval()

    inference_time = 0
    num_imgs = 0

    num_samples = len(test_loader)
    print('=> %d samples found in the test set' % num_samples)

    epess = 0
    areas = 0
    dist_errss = 0

    wandb.watch(aanet)

    csvHeader = ['No.', 'x_min', 'x_max', 'y_min', 'y_max','GT_disp', 'PredDisp', 'EPE', 'GT_depth', 'PredDepth', 'DepthErr']

    for i, sample in enumerate(test_loader):
        if args.count_time and i == args.num_images:  # testing time only
            break

        csvFileName = sample['left_name'][0][:-4] + '_12x12.csv'
        csvFileName = os.path.join(args.output_dir, csvFileName)
        
        with open(csvFileName, 'w', newline='') as file:
            csvwriter = csv.writer(file)
            csvwriter.writerow(csvHeader)

        epes = 0
        area = 0
        dist_errs = 0

        # ## Original code 
        # if i % 100 == 0:
        #     print('=> Inferencing %d/%d' % (i, num_samples))

        print('=> Inferencing %d/%d' % (i, num_samples))

        '''
        ## Test full image
        left_img = sample['left'].to(device)
        right_img = sample['right'].to(device)
        gt_disp_img = sample['disp'].to(device)

        infer_time = 0

        with torch.no_grad():
            time_start = time.perf_counter()
            pred_disp_full = aanet(left_img, right_img)[-1]  # [B, H, W]
            infer_time += time.perf_counter() - time_start
        
        print('Full image inference time: ', infer_time)

        disp_pred_full = pred_disp_full.detach().cpu().numpy()  # [H, W]
        save_name_pred = str(j) + '_pred_' + sample['left_name'][b]
        save_name_pred = os.path.join(args.output_dir, save_name_pred)
        skimage.io.imsave(save_name_pred, (disp_pred_full * 256.).astype(np.uint16))

        infer_time = 0
        '''

        print("No. of bbox: ", len(sample['left_bboxes']))
        if len(sample['left_bboxes']) == 0: continue
        for j, bbox in enumerate(sample['left_bboxes']):
            ## bbox: [<class>, <x_min>, <y_min>, <x_max>, <y_max>]

            x_min = bbox[1]
            x_max = bbox[3]
            y_min = bbox[2]
            y_max = bbox[4]
            print('BBox: x_min = {}, x_max = {}, y_min = {}, y_max = {}'.format(x_min, x_max, y_min, y_max))

            ## eliminate objects near left and right margins
            if (x_min < 192) or (x_max > 2938): continue

            ## Eliminate objects too far
            if (y_min  < 96): continue

            crop_width = x_max - x_min
            crop_height = y_max - y_min

            if (crop_width < 30) and (crop_height < 30): continue ## filter small objects

            x_min_p = x_min - 96

            x_max_p = x_max + (96-crop_width%96)
            y_min_p = y_min - (96-crop_height%96)

            left = sample['left'][:,:,y_min_p:y_max,x_min_p:x_max_p].to(device)  # [B, 3, H, W]
            right = sample['right'][:,:,y_min_p:y_max,x_min_p:x_max_p].to(device)
            gt_disp = sample['disp'][:,y_min:y_max,x_min:x_max].to(device)

            print('Predict region: width = {}, height = {}'.format(left.shape[2], left.shape[3]))

            # Pad
            # ori_height, ori_width = left.size()[2:]
            ori_height, ori_width = sample['left'].to(device).size()[2:]
            if ori_height < args.img_height or ori_width < args.img_width:
                top_pad = args.img_height - ori_height
                right_pad = args.img_width - ori_width

                # Pad size: (left_pad, right_pad, top_pad, bottom_pad)
                left = F.pad(left, (0, right_pad, top_pad, 0))
                right = F.pad(right, (0, right_pad, top_pad, 0))

            # Warpup
            if i == 0 and args.count_time:
                with torch.no_grad():
                    for _ in range(10):
                        aanet(left, right)

            num_imgs += left.size(0)

            with torch.no_grad():
                time_start = time.perf_counter()
                pred_disp = aanet(left, right)[-1]  # [B, H, W]
                inference_time += time.perf_counter() - time_start

            if pred_disp.size(-1) < left.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, (left.size(-2), left.size(-1)),
                                        mode='bilinear') * (left.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            # Crop
            if ori_height < args.img_height or ori_width < args.img_width:
                if right_pad != 0:
                    pred_disp = pred_disp[:, top_pad:, :-right_pad]
                else:
                    pred_disp = pred_disp[:, top_pad:]

            offset_x = int(crop_width/2-6)
            if offset_x <=0: offset_x = 1
            # offset = 1
            x_min_bb = 96 + offset_x
            x_max_bb = x_max_p - x_min_p - (96-crop_width%96) - offset_x

            offset_y = int(crop_height/2-6)
            if offset_y <=0:  offset_y = 1
            y_min_bb = (96-crop_height%96) + offset_y
            pred_disp_bb = pred_disp[:, y_min_bb:-offset_y, x_min_bb:x_max_bb]
            
            gt_disp = gt_disp[:,offset_y:-offset_y, offset_x:-offset_x]
            # if i >=38:
            #     pdb.set_trace()

            # print('Mean disparity of predicted bbox: ', pred_disp_bb.mean())
            print('ROI shape: ', pred_disp_bb.shape)

            mask = (gt_disp > 0) & (gt_disp < args.max_disp)
            # 3-pixel error
            thres3 = thres_metric(pred_disp_bb, gt_disp, mask, 3.0)
            print('3-pixel error: ', thres3)

            # EPE 
            epe = F.l1_loss(gt_disp[mask], pred_disp_bb[mask], reduction='mean')
            epes += epe*(x_max - x_min_bb)*(y_max-y_min_bb)
            area += (x_max - x_min_bb)*(y_max-y_min_bb)

            # d1 = d1_metric(pred_disp, gt_disp, mask)
            print('EPE: ', epe)

            ## Calculate depth error
            gt_depth = Sys.disp2depth(gt_disp.detach().cpu().numpy())
            pred_depth = Sys.disp2depth(pred_disp_bb.detach().cpu().numpy())

            depth_err = abs(gt_depth - pred_depth)
            

            # ['No.', 'x_min', 'x_max', 'y_min', 'y_max','GT_disp', 'PredDisp', 'EPE', 'GT_depth', 'PredDepth', 'DepthErr']
            data = [j, x_min, x_max, y_min, y_max, gt_disp.mean(), pred_disp_bb.mean(), epe, gt_depth.mean(), pred_depth.mean(), depth_err.mean()]
            with open(csvFileName, 'a', newline='') as file:
                csvwriter = csv.writer(file)
                csvwriter.writerow(data)
            wandb.log({'id': j,'GT_disp': gt_disp.mean(), 'PredDisp': pred_disp_bb.mean(), 'EPE': epe, 'GT_depth': gt_depth.mean(), 'PredDepth': pred_depth.mean(), 'DepthErr': depth_err.mean()})
            


            for b in range(pred_disp.size(0)):
                ## Original code
                # disp = pred_disp[b].detach().cpu().numpy()  # [H, W]
                # save_name = str(j) + '_' + sample['left_name'][b]
                # save_name = os.path.join(args.output_dir, save_name)

                ## Pred_disp
                disp_pred = pred_disp_bb[b].detach().cpu().numpy()  # [H, W]
                # disp_pred = pred_disp[b].detach().cpu().numpy()  # [H, W] # full predicted region
                save_name_pred = sample['left_name'][b][:-4] + '_pred_' + str(j) + '.png'
                save_name_pred = os.path.join(args.output_dir, save_name_pred)

                ## GT disp
                disp_gt = gt_disp[b].detach().cpu().numpy()  # [H, W]
                save_name_gt = sample['left_name'][b][:-4] + '_gt_' + str(j) + '.png'
                save_name_gt = os.path.join(args.output_dir, save_name_gt)

                # ## Cropped left image
                left_imge = Image.fromarray(left[0].permute(1,2,0).detach().cpu().numpy().astype('uint16'),'RGB')
                save_name_left = sample['left_name'][b][:-4] + '_left_' + str(j) + '.png'
                save_name_left = os.path.join(args.output_dir, save_name_left)

                # utils.check_path(os.path.dirname(save_name))
                utils.check_path(os.path.dirname(save_name_pred))
                utils.check_path(os.path.dirname(save_name_gt))
                if not args.count_time:
                    if args.save_type == 'pfm':
                        if args.visualize:
                            # skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))
                            skimage.io.imsave(save_name_pred, (disp_pred * 256.).astype(np.uint16))
                            skimage.io.imsave(save_name_gt, (disp_gt * 256.).astype(np.uint16))

                        ## Original code
                        # save_name = save_name[:-3] + 'pfm'
                        # write_pfm(save_name, disp)

                        ## Pred
                        save_name_pred = save_name_pred[:-3] + 'pfm'
                        write_pfm(save_name_pred, disp_pred)
                        ## GT
                        save_name_gt = save_name_gt[:-3] + 'pfm'
                        write_pfm(save_name_gt, disp_gt)
                    elif args.save_type == 'npy':
                        ## Original code
                        # save_name = save_name[:-3] + 'npy'
                        # np.save(save_name, disp)

                        ## Pred
                        save_name_pred = save_name_pred[:-3] + 'npy'
                        np.save(save_name_pred, disp_pred)

                        save_name_gt = save_name_gt[:-3] + 'npy'
                        np.save(save_name_gt, disp_gt)
                    else:
                        # skimage.io.imsave(save_name, (disp * 256.).astype(np.uint16))
                        skimage.io.imsave(save_name_pred, (disp_pred * 256.).astype(np.uint16))
                        skimage.io.imsave(save_name_gt, (disp_gt * 256.).astype(np.uint16))
                        left_imge.save(save_name_left)

                # Distance error
                dist_error = dist_err((disp_pred * 256.).astype(np.uint16), (disp_gt* 256.).astype(np.uint16), mask.detach().cpu().numpy())
                dist_errs += dist_error*(x_max - x_min_bb)*(y_max-y_min_bb)
                print('Distance error: ', dist_error)

        epess += epes
        areas += area
        dist_errss += dist_errs
        if area == 0: continue
        print('==> Image Avg EPE: ', epes/area)
        print('==> Image Avg Distance error: ', dist_errs/area)
        wandb.log({'img_number': i,'Avg EPE': epes/area, 'Avg Distance error': dist_errs/area})

    print('===> Mean inference time for %d images: %.3fs' % (num_imgs, inference_time / num_imgs))
    print('===> Avg EPE: ', epess/areas)
    print('===> Avg Distance error: ', dist_errss/areas)

if __name__ == '__main__':
    main()
