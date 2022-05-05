import torch
import numpy as np

EPSILON = 1e-8


def epe_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        epe = np.mean(np.abs(d_est - d_gt))
    else:
        epe = torch.mean(torch.abs(d_est - d_gt))

    return epe


def d1_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = (e > 3) & (e / d_gt > 0.05)

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean

def disp2depth(disp_img):
    ## Apolloscape camera parameters
    focal_x = 2301.3147
    baseline = 0.622

    depth = torch.zeros(disp_img.shape)
    depth = depth + (focal_x * baseline)
    # disp_img = disp_img/256
    depth = torch.div(depth, disp_img)

    return depth
    

def dist_err(d_est, d_gt, mask):
    depth_est = disp2depth(d_est[mask])
    depth_gt  = disp2depth(d_gt[mask])
    depth_err = torch.abs(depth_est-depth_gt)
    mean = torch.mean(depth_err)
    return mean