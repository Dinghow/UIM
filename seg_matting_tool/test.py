#coding=utf-8
import numpy as np
import os.path
import logging
import argparse
import cv2
import torch.nn.parallel
import numpy as np
from PIL import Image
import util.helpers as helpers
from util import dataset
from util.util import AverageMeter, compute_mse, compute_sad, compute_gradient, compute_connectivity, get_cuda_devices, get_unknown_tensor_from_pred
from torch.nn.functional import upsample
import os
import random
import time
import cv2
import numpy as np
import logging
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
#import apex
from torchvision import transforms
from tensorboardX import SummaryWriter
from util.custom_transforms import interactiveMattingTransform
from util import dataset, config, helpers


def sort_dict(dict_src):
    dict_new = {}
    for k in sorted(dict_src):
        dict_new.update({k: dict_src[k]})
    return dict_new

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ade20k/ade20k_pspnet50.yaml', help='config file')
    parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_relax_pad(relax_pad, extreme_points):
    if relax_pad <= 0:
        return 0
    if relax_pad >= 1:
        return int(relax_pad)

    x_min, y_min = np.min(extreme_points, axis=0)
    x_max, y_max = np.max(extreme_points, axis=0)
    x_len = x_max - x_min + 1
    y_len = y_max - y_min + 1
    return max(20, int(relax_pad * max(x_len, y_len)))

def main():
    global args, logger, writer
    use_void_pixels=True
    logger = get_logger()
    args = get_parser()
    # writer = SummaryWriter(args.save_folder)
    if args.test_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)
    else:
        args.test_gpu = get_cuda_devices()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(args)# 在屏幕上打印信息
    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    # transform and dataloader
    _interactive_matting_transform = interactiveMattingTransform(channel=args.in_channels, no_crop=args.no_crop, relax_crop=args.relax_crop,\
                                    use_iogpoints=args.use_iogpoints, use_roimasking=args.use_roimasking, use_trimap=args.use_trimap,\
                                    use_in_point=args.use_in_point, use_bbox=args.use_bbox, use_iogdextr=args.use_iogdextr, use_extreme_points=args.use_extreme_points, use_scribble=args.use_scribble,\
                                    rotate_degree=args.rotate_degree, scale=args.scale, shear=args.shear,\
                                    flip=args.flip, crop_size=args.crop_size, mask_type=args.mask_type, bbox_type=args.bbox_type)
    composed_transforms_ts = _interactive_matting_transform.getTestTransform()

    val_data = dataset.Composition1KMatting(root=args.data_root, split=args.test_split,transform=composed_transforms_ts, task=args.task, num_bgs=args.test_num_bgs)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_test, shuffle=False, num_workers=args.workers_test, pin_memory=True, sampler=None)

    # model
    if args.arch == 'uim':
        from model.mattingnet import Unified_Interactive_Matting
        model = Unified_Interactive_Matting(n_classes=args.classes, in_channels=args.in_channels, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, fusion_method=args.fusion_method)
    elif args.arch == 'uim_trimap':
        from model.mattingnet import Unified_Interactive_Matting_trimap
        model = Unified_Interactive_Matting_trimap(n_classes=args.classes, in_channels=args.in_channels, encoder_layers=args.encoder_layers, decoder_layers=args.decoder_layers, fusion_method=args.fusion_method)
    else:
        raise RuntimeError('Wrong arch.')

    logger.info(model)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model = model.to(device)
    model.eval()

    # checkpoint
    model_path = args.model_path
    if os.path.isfile(model_path):
        logger.info("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        logger.info("=> loaded checkpoint '{}'".format(model_path))
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(model_path))
    
    # evaluate
    print('evaluating Network')
    eval_result = dict()
    eval_result['all_mse'] = AverageMeter()
    eval_result['all_sad'] = AverageMeter()
    eval_result['all_grad'] = AverageMeter()
    eval_result['all_connectivity'] = AverageMeter()
    eval_result['all_mse_tri_free'] = AverageMeter()
    eval_result['all_sad_tri_free'] = AverageMeter()
    eval_result['all_grad_tri_free'] = AverageMeter()
    eval_result['all_connectivity_tri_free'] = AverageMeter()    

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    with torch.no_grad():
        # Main Testing Loop
        for ii, sample in enumerate(val_loader):
            if ii % 10 == 0:
                print('Evaluating: {} of {} batches'.format(ii, len(val_loader)))
            # predict result and gt
            image = sample['image']
            alpha = sample['alpha']
            trimap_ori = sample['trimap_ori']
            alpha_shape = sample['alpha_shape']
            alpha_ori = sample['alpha_ori']
            metas = sample["meta"]
            if args.use_iogpoints:
                interactive = torch.cat((sample['in_points'],sample['out_points']), dim=1)
            elif args.use_trimap:
                interactive = sample['trimap']
            elif args.use_bbox:
                interactive = sample['out_points']
            elif args.use_in_point:
                interactive = sample['in_points']
            elif args.use_extreme_points:
                interactive = sample['extreme_points']
            elif args.use_scribble:
                interactive = sample['scribble']
            else:
                interactive = None
            pred = model.forward(image, interactive)
            pred = pred.to(torch.device('cpu'))
            
            alpha = helpers.tens2image(alpha.squeeze())
            image = helpers.tens2image(image)
            trimap_ori = helpers.tens2image(trimap_ori)
            alpha_ori = helpers.tens2image(alpha_ori)
            
            pred = helpers.tens2image(pred.squeeze())
            h, w = alpha_shape
            pred = pred[:h, :w]

            # Restore the image to its original size
            if not args.no_crop:
                h = metas['im_size'][0][0].item()
                w =  metas['im_size'][1][0].item()
                boundary = [metas['boundary'][0][0].item(), metas['boundary'][1][0].item(), 
                            metas['boundary'][2][0].item(), metas['boundary'][3][0].item()]
                points= np.array([[boundary[2], boundary[0]], 
                                    [boundary[2], boundary[1]], 
                                    [boundary[3], boundary[0]], 
                                    [boundary[3], boundary[1]]])
                relax_pad = get_relax_pad(args.relax_crop, points)
                pred = helpers.align2fullmask(pred, (h,w), points, relax=relax_pad)
                alpha = helpers.align2fullmask(alpha, (h,w), points, relax=relax_pad)
                assert (alpha == alpha_ori).all()
                
            pred_tri_free = np.copy(pred)
            pred[trimap_ori == 0] = 0.0
            pred[trimap_ori == 255] = 1.0

            if args.save_pic:
                # interactive = helpers.tens2image(interactive)
                # get object id
                pic_id = sample["meta"]["image"][0]
                alpha_dir = os.path.join(args.save_folder, pic_id + '_alpha.png')
                pred_dir = os.path.join(args.save_folder, pic_id + '_pred.png')
                pred_tri_free_dir = os.path.join(args.save_folder, pic_id + '_pred_tri_free.png')
                # interactive_dir = os.path.join(args.save_folder, pic_id + '_interactive.png')
                if args.use_trimap:
                    trimap_dir = os.path.join(args.save_folder, pic_id + '_trimap.png')
                    cv2.imwrite(trimap_dir, trimap_ori)
                cv2.imwrite(pred_dir, pred * 255)
                cv2.imwrite(pred_tri_free_dir, pred_tri_free * 255)
                cv2.imwrite(alpha_dir, alpha * 255)
                # cv2.imwrite(interactive_dir, interactive * 255)

            # Evaluate
            # trimap-free matting metric method by generated trimap, referenced from BackgroundMatting V2.0
            mse = compute_mse(pred, alpha, trimap_ori)
            sad = compute_sad(pred, alpha, trimap_ori)
            grad = compute_gradient(pred, alpha, trimap_ori)
            connectivity = compute_connectivity(pred, alpha, trimap_ori)
            # trimap-free matting metric method by compute all gt area
            mse_tri_free = compute_mse(pred_tri_free, alpha)
            sad_tri_free = compute_sad(pred_tri_free, alpha)
            grad_tri_free = compute_gradient(pred_tri_free, alpha)
            connectivity_tri_free = compute_connectivity(pred_tri_free, alpha)

            eval_result['all_mse'].update(mse)
            eval_result['all_sad'].update(sad)
            eval_result['all_grad'].update(grad)
            eval_result['all_connectivity'].update(connectivity)
            eval_result['all_mse_tri_free'].update(mse_tri_free)
            eval_result['all_sad_tri_free'].update(sad_tri_free)
            eval_result['all_grad_tri_free'].update(grad_tri_free)
            eval_result['all_connectivity_tri_free'].update(connectivity_tri_free)
                
    logger.info("=========result==========")
    logger.info("MSE: ")
    logger.info(eval_result['all_mse'].avg)
    logger.info("SAD: ")
    logger.info(eval_result['all_sad'].avg)
    logger.info("Grad: ")
    logger.info(eval_result['all_grad'].avg)
    logger.info("Connectivity: ")
    logger.info(eval_result['all_connectivity'].avg)
    logger.info("MSE trimap-free: ")
    logger.info(eval_result['all_mse_tri_free'].avg)
    logger.info("SAD trimap-free: ")
    logger.info(eval_result['all_sad_tri_free'].avg)
    logger.info("Grad trimap-free: ")
    logger.info(eval_result['all_grad_tri_free'].avg)
    logger.info("Connectivity trimap-free: ")
    logger.info(eval_result['all_connectivity_tri_free'].avg)

if __name__ == '__main__':
    main()