import os
import numpy as np
import PIL
from PIL import Image

import torch
from torch import nn
import torch.nn.init as initer
import torch.nn.functional as F
import cv2
import heapq
import scipy.ndimage
from skimage.measure import label
import scipy.ndimage.morphology
try:    
    import kornia
except:
    print('kornia not installed')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def step_learning_rate(base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    return lr


def poly_learning_rate(base_lr, curr_iter, max_iter, power=0.9):
    """poly learning rate policy"""
    lr = base_lr * (1 - float(curr_iter) / max_iter) ** power
    return lr


def triangle_learning_rate(base_lr, tmp_iter, triangular_step=0.5, restart_period = 50):
    """
    Sets the learning rate to triangle
    restart_period means the width of the triangle
    """
    t_cur = tmp_iter % restart_period
    inflection_point = triangular_step * restart_period
    point_of_triangle = (t_cur / inflection_point
                             if t_cur < inflection_point
                             else 1.0 - (t_cur - inflection_point)
                             / (restart_period - inflection_point))
    return (point_of_triangle + 0.1) * base_lr


def exponential_learning_rate(base_lr, tmp_iter, iters, gamma=0.9):
    """Sets the learning rate to Exponential decay"""
    lr = base_lr * (gamma ** (tmp_iter / (iters / 50)))
    return lr


def exponential_triangle_lr(base_lr, tmp_iter, iters = 100, restart_period = 50):
    base_lr_1 = exponential_learning_rate(base_lr = base_lr, tmp_iter = tmp_iter, iters = iters)
    base_lr_2 = triangle_learning_rate(base_lr_1, tmp_iter = tmp_iter, restart_period = restart_period)
    return base_lr_2


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def init_weights(model, conv='kaiming', batchnorm='constant', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.modules.conv._ConvNd)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0.0)

        elif isinstance(m, (nn.modules.batchnorm._BatchNorm)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)


def group_weight(weight_group, module, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


# alpha prediction loss: the abosolute difference between the ground truth alpha values and the
# predicted alpha values at each pixel. However, due to the non-differentiable property of
# absolute values, we use the following loss function to approximate it.
def alpha_prediction_loss(y_pred, y_true):
    h, w = y_pred.shape[-2:]
    y_true = y_true[:, :2, ...]
    y_pred = y_pred.reshape((-1, 1, h*w))
    y_true = y_true.reshape((-1, 2, h*w))
    mask = y_true[:, 1, :]
    diff = y_pred[:, 0, :] - y_true[:, 0, :]
    diff = diff * mask
    num_pixels = torch.sum(mask)
    return torch.sum(torch.sqrt(torch.pow(diff, 2) + 1e-12)) / (num_pixels + 1e-6)


def alpha_prediction_loss_trimap_free(y_pred, y_true):
    h, w = y_pred.shape[-2:]
    y_pred = y_pred.reshape((-1, 1, h*w))
    y_true = y_true.reshape((-1, 1, h*w))
    diff = y_pred[:, 0, :] - y_true[:, 0, :]
    num_pixels = torch.sum(y_pred[:, 0, :])
    return torch.sum(torch.sqrt(torch.pow(diff, 2) + 1e-12)) / (num_pixels + 1e-6)


def gradient_loss_trimap_free(y_pred, y_true):
    return F.l1_loss(kornia.sobel(y_pred), kornia.sobel(y_true))


def mask_alpha_prediction_loss(y_pred, y_true, mask):
    h, w = y_pred.shape[-2:]
    y_pred = y_pred.reshape((-1, 1, h*w))
    y_true = y_true.reshape((-1, 1, h*w))
    mask = mask.reshape((-1, 1, h*w))
    diff = y_pred[:, 0, :] - y_true[:, 0, :]
    diff = diff * mask
    num_pixels = torch.sum(mask)
    return torch.sum(torch.sqrt(torch.pow(diff, 2) + 1e-12)) / (num_pixels + 1e-6)


def deep_matting_loss(y_pred, img, fg, bg, y_true):
    h, w = y_pred.shape[-2:]
    alpha_loss = alpha_prediction_loss(y_pred, y_true)
    mask = y_true[:, 1, ...].reshape((-1, 1, h, w))
    num_pixels = torch.sum(mask)
    tri_mask = torch.cat((mask, mask, mask), dim=1)
    tri_pred = torch.cat((y_pred, y_pred, y_pred), dim=1)
    new_img = tri_pred * fg + (1 - tri_pred) * bg
    composite_loss = torch.sqrt(torch.pow(new_img-img, 2) + 1e-12) / 255.0
    composite_loss = torch.sum(tri_mask * composite_loss) / (num_pixels + 1e-6) / 3.0
    return 0.5 * alpha_loss + 0.5 * composite_loss


def composition_loss_trimap_free(y_pred, img, fgs, bgs, y_true):
    h, w = y_pred.shape[-2:]
    # alpha_loss = alpha_prediction_loss_trimap_free(y_pred, y_true)
    num_pixels = torch.sum(y_true[:, 0, ...])
    tri_pred = torch.cat((y_pred, y_pred, y_pred), dim=1)
    new_img = tri_pred * fgs + (1 - tri_pred) * bgs
    composite_loss = torch.sqrt(torch.pow(new_img-img, 2) + 1e-12) / 255.0
    composite_loss = torch.sum(composite_loss) / (num_pixels + 1e-6) / 3.0
    ### Test composition loss
    # image_show_path = "image_show"
    # if not os.path.exists(image_show_path):
    #     os.makedirs(image_show_path)
    # for j in range(5):
    #     print("***********************************************")
    #     image = img[j, :3, :, :]
    #     image = image.cpu().numpy()
    #     image = image.astype("uint8")
    #     image = np.transpose(image, [1, 2, 0])
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #     fg = fgs[j, :3, :, :]
    #     print('fg:', fg.shape)
    #     fg = fg.cpu().numpy().astype('uint8')
    #     fg = np.transpose(fg, [1, 2, 0])
    #     fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)   
    #     bg = bgs[j, :3, :, :]
    #     bg = bg.cpu().numpy().astype('uint8')
    #     bg = np.transpose(bg, [1, 2, 0])
    #     bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    #     new_im = new_img[j, :3, :, :]
    #     print('new_img:', new_img.shape)
    #     new_im = new_im.detach().cpu().numpy().astype('uint8')
    #     new_im = np.transpose(new_im, [1, 2, 0])
    #     new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2BGR) 
    #     cv2.imwrite(image_show_path + "/{}_image.png".format(j), image)
    #     cv2.imwrite(image_show_path + "/{}_fg.png".format(j), fg)
    #     cv2.imwrite(image_show_path + "/{}_bg.png".format(j), bg)
    #     cv2.imwrite(image_show_path + "/{}_new_img.png".format(j), new_im)
    # return 0.5 * alpha_loss + 0.5 * composite_loss
    return composite_loss


def end2end_deep_matting_loss(y_pred1, y_pred2, img, fg, bg, y_true):
    loss1 = deep_matting_loss(y_pred1, img, fg, bg, y_true)
    loss2 = alpha_prediction_loss(y_pred2, y_true)
    return loss1 + loss2


def adapation_matting_loss(alpha_out, classify_out, y_true):
    classes = torch.argmax(classify_out, dim=1)
    classes[classes==2] = 0
    prediction_loss = mask_alpha_prediction_loss(alpha_out, y_true[:, :1, ...], classes)
    classification_loss = torch.nn.functional.cross_entropy(classify_out, y_true[:, 2, ...].long())
    return prediction_loss + 0.5*classification_loss


def adapation_matting_loss_uncertain(alpha_out, classify_out, y_true):
    classes = torch.argmax(classify_out, dim=1)
    classes[classes==2] = 0
    prediction_loss = mask_alpha_prediction_loss(alpha_out, y_true[:, :1, ...], classes)
    classification_loss = torch.nn.functional.cross_entropy(classify_out, y_true[:, 2, ...].long())
    return prediction_loss, classification_loss

def adapation_matting_compo_loss(y_pred, classify_out, img, fg, bg, y_true): 
    classes = torch.argmax(classify_out, dim=1)
    classes[classes==2] = 0
    alpha_loss = mask_alpha_prediction_loss(y_pred, y_true[:, :1, ...], classes)
    mask = classes.reshape((-1,1,320,320))
    #mask = y_true[:, 1, ...].reshape((-1, 1, 320, 320))
    num_pixels = torch.sum(mask)
    tri_mask = torch.cat((mask, mask, mask), dim=1)
    tri_pred = torch.cat((y_pred, y_pred, y_pred), dim=1)
    new_img = tri_pred * fg + (1 - tri_pred) * bg
    composite_loss = torch.sqrt(torch.pow(new_img-img, 2) + 1e-12) / 255.0
    composite_loss = torch.sum(tri_mask * composite_loss) / (num_pixels + 1e-6) / 3.0
    classification_loss = torch.nn.functional.cross_entropy(classify_out, y_true[:, 2, ...].long())
    return 0.4 * alpha_loss + 0.3 * classification_loss + 0.3 * composite_loss

def ada_end2end_deep_matting_loss(y_pred1, y_pred2, classify_out, img, fg, bg, y_true):
    loss1 = adapation_matting_loss(y_pred1, classify_out, y_true)
    loss2 = alpha_prediction_loss(y_pred2, y_true)
    return loss1 + loss2

def compute_accuracy(tri,alpha):
    h, w = tri.shape[:2]
    tri_gt = np.ones((h, w), dtype=np.float32)
    tri_gt[alpha == 0] = 0
    tri_gt[alpha == 1.0] = 2
    right_pixel = np.sum(tri_gt==tri)
    return right_pixel/(h*w)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


# compute the MSE error given a prediction, a ground truth and a trimap.
# pred: the predicted alpha matte
# target: the ground truth alpha matte
# trimap: the given trimap
def compute_mse(pred, alpha, trimap=None):
    if trimap is not None:
        num_pixels = float((trimap == 128).sum())
        return ((pred - alpha) ** 2 * (trimap == 128) ).sum() / (num_pixels + 1e-8)
    else:
        num_pixels = float(np.prod(alpha.shape))
        return ((pred - alpha) ** 2).sum() / (num_pixels + 1e-8)


# compute the SAD error given a prediction and a ground truth.
def compute_sad(pred, alpha, trimap=None):
    diff = np.abs(pred - alpha)
    if trimap is not None:
        return np.sum(diff * (trimap == 128)) / 1000
    else:
        return np.sum(diff) / 1000


# compute the Gradient error
def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(np.int32)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy


def compute_gradient(pred, target, trimap=None):
    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    if trimap is not None:
        loss = np.sum(error_map[trimap == 128])
    else:
        loss = np.sum(error_map)

    return loss / 1000.


# compute the Connectivity error
def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def compute_connectivity(pred, target, trimap=None, step=0.1):
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=np.float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(np.int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(np.int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(np.int)
        flag = ((l_map == -1) & (omega == 0)).astype(np.int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(np.int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(np.int)
    if trimap is not None:
        loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])
    else:
        loss = np.sum(np.abs(pred_phi - target_phi))

    return loss / 1000.


def compute_metric(pred, gt, args):
    h, w = pred.shape[:2]
    if np.max(pred) == 1:
        pred = pred*255

    if args.alpha_threshold is None:
        args.alpha_threshold = 128
    elif args.alpha_threshold < 0 or args.alpha_threshold > 255:
        args.alpha_threshold = 128
    pred[pred >= args.alpha_threshold] = 255
    pred[pred < args.alpha_threshold]  = 0
    gt[gt > 0] = 255

    front_pred = (pred == 255)
    front_gt = (gt == 255)
    compare = pred[front_pred == front_gt]
    intersection = sum(compare == 255)
    union = h*w - sum(compare == 0)
    pred_mask = sum(sum(front_pred))
    gt_mask = sum(sum(front_gt))

    iou = float(intersection) / (union + 1)
    acc = float(intersection) / (pred_mask + 1)
    recall = float(intersection) / (gt_mask + 1)

    pred = pred.astype(np.uint8)
    gt = gt.astype(np.uint8)
    contour_pred = cv2.Canny(np.array(pred), 127, 255)
    contour_gt = cv2.Canny(np.array(gt), 127, 255)
    front_pred = (contour_pred == 255)
    front_gt = (contour_gt == 255)
    compare = contour_pred[front_pred == front_gt]
    intersection = sum(compare == 255)
    pred_mask = sum(sum(front_pred))
    ctacc = float(intersection) / (pred_mask + 1)
    return iou, acc, recall, ctacc


def visualization_result(pred, gt, args, name):
    h, w = pred.shape[:2]
    if np.max(pred) == 1:
        pred = pred * 255
    # save alpha result
    cv2.imwrite(os.path.join(args.save_folder, args.trimap, 'matting', name+'.png'), pred)

    if args.alpha_threshold is None:
        args.alpha_threshold = 128
    elif args.alpha_threshold < 0 or args.alpha_threshold > 255:
        args.alpha_threshold = 128
    pred[pred >= args.alpha_threshold] = 255
    pred[pred < args.alpha_threshold]  = 0
    gt[gt > 0] = 255

    # plot contours and polygons with Canny or findContours
    pred = pred.astype(np.uint8)
    contourimg = np.zeros((h, w, 3), np.uint8)
    contourimg[gt==255, :] = 255
    ret, thresh = cv2.threshold(pred, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 2, 1)
    epsilon = max(cv2.arcLength(contours[0], True) / 800.0, 7)
    thickness = min(int(sum(sum(pred / 255)) * 3e-6), 2)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(contourimg, cnt, -1, (0, 0, 255), thickness)
        cv2.drawContours(contourimg, approx, -1, (255, 0, 0), thickness*3)
    #contour_pred = cv2.Canny(np.array(pred.astype(np.uint8)), 100, 200)

    # save visualization result
    cv2.imwrite(os.path.join(args.save_folder, args.trimap, 'contour', name+'.png'), contourimg)
    cv2.imwrite(os.path.join(args.save_folder, args.trimap, 'segmentation', name+'.png'), pred)


def get_cuda_devices():
    os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp')
    memory_gpu = np.array([int(x.split()[2]) for x in open('tmp', 'r').readlines()])
    clen = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    devices = heapq.nlargest(clen, range(len(memory_gpu)), memory_gpu.take)
    test_gpu = devices
    os.environ['CUDA_VISIBLE_DEVICES']=str(devices)[1:-1]
    os.system('rm tmp')
    return devices


def error_map_loss(err_pred, alpha_pred, alpha_gt):
    err_gt = torch.abs(alpha_pred.detach() - alpha_gt)
    return F.mse_loss(err_pred, err_gt)


def weighted_mse_loss(weight_map, pred, gt, batch_average=False, size_average=True):
    """Computer the mean square loss with a weight matrix
    Args:
    weight_map: weight matrix, (B, 1, H, W)
    pred: the prediction of foreground, (B, 1, H, W)
    gt: the ground truth of foreground, (B, 1, H, W)
    batch_average: divide batch size
    size_average: divide number of pixels
    Returns:
    Weighted MSE loss of the error map
    """
    assert(gt.size() == pred.size())
    assert(pred.size() == weight_map.size())

    loss_val = 0.5*(pred - gt)**2
    loss_val = torch.mul(weight_map, loss_val)

    if batch_average:
        final_loss = loss_val.sum() / gt.size()[0]

    elif size_average:
        final_loss = loss_val.sum() / np.prod(gt.size())

    else:
        raise ValueError("Please set the loss type.\n")

    return final_loss


def cross_entropy_loss(output, label, size_average=True, batch_average=False):
    assert(output.size() == label.size())
    labels = torch.ge(label, 0.5).float()
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero))) 
    final_loss = torch.sum(-loss_val)
    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]
    return final_loss


def class_cross_entropy_loss(output, label, size_average=True, batch_average=False, void_pixels=None):
    assert(output.size() == label.size())
    labels = torch.ge(label, 0.5).float()
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        final_loss = torch.mul(w_void, loss_val)
    else:
        final_loss=loss_val     
    final_loss = torch.sum(-final_loss)
    if size_average:
        if void_pixels is not None:
            final_loss /= w_void.sum()
        else:
            final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]
    return final_loss

class CE_LOSS(torch.nn.Module):
    def __init__(self):
        super(CE_LOSS, self).__init__()

    def forward(self, pred, target, size_average=True, batch_average=False, void_pixels=None):
        return class_cross_entropy_loss(pred, target, size_average=size_average, batch_average=batch_average, void_pixels=void_pixels)


def weighted_class_cross_entropy_loss(output, label, size_average=True, batch_average=False, void_pixels=None, weight=None):
    assert(output.size() == label.size())
    labels = torch.ge(label, 0.5).float()
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        final_loss = torch.mul(w_void, loss_val)
    else:
        final_loss=loss_val   
    if weight is not None:
        final_loss = torch.mul(final_loss, weight)
    else:
        final_loss = final_loss
    final_loss = torch.sum(-final_loss)
    if size_average:
        #final_loss /= np.prod(label.size())
        final_loss /= w_void.sum()
    elif batch_average:
        final_loss /= label.size()[0]
    return final_loss

class WCE_LOSS(torch.nn.Module):
    def __init__(self):
        super(WCE_LOSS, self).__init__()

    def forward(self, pred, target, size_average=True, batch_average=False, void_pixels=None, weight=None):
        return weighted_class_cross_entropy_loss(pred, target, size_average=size_average, batch_average=batch_average, void_pixels=void_pixels, weight=weight)


def select_focus_error_region(err, sampling_mode='sampling', select_ratio=1.0):
    """Select error region to calculate loss from error map
    Input:
        err： error map （B, 1, H, W）
        sampling_mode: 'sampling', 'thresholding'. 'sampling' select top k pixels sorted by error value. 'thresholding' get all the pixels after thresholding.
    Return:
        reg: focus regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
    """
    if sampling_mode == 'sampling':
        # Sampling mode.
        b, _, h, w = err.shape
        num_sampling_pixels = int(h * w * select_ratio)
        err = err.view(b, -1)
        idx = err.topk(num_sampling_pixels, dim=1, sorted=False).indices
        reg = torch.zeros_like(err)
        reg.scatter_(1, idx, 1.)
        reg = reg.view(b, 1, h, w)
    else:
        # Thresholding mode.
        threshold = 0 # 0.5?
        reg = err.gt(threshold).float()
    return reg

def class_cross_entropy_loss_v1(output, label, size_average=True, batch_average=False, void_pixels=None, select_ratio=1.0):
    assert(output.size() == label.size())
    labels = torch.ge(label, 0.5).float()
    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))
    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        final_loss = torch.mul(w_void, loss_val)
    else:
        final_loss=loss_val     
    final_loss = -final_loss
    select_reg = select_focus_error_region(final_loss, select_ratio=select_ratio)
    final_loss = torch.mul(select_reg, final_loss)
    final_loss = torch.sum(final_loss)
    if size_average:
        #final_loss /= np.prod(label.size())
        final_loss /= select_reg.sum()
    elif batch_average:
        final_loss /= label.size()[0]
    return final_loss


def class_balanced_cross_entropy_loss(output, label, size_average=False, batch_average=True, void_pixels=None):
    """Define the class balanced cross entropy loss to train the network
    Args:
    output: Output of the network
    label: Ground truth label
    size_average: return per-element (pixel) average loss
    batch_average: return per-batch average loss
    void_pixels: pixels to ignore from the loss
    Returns:
    Tensor that evaluates the loss
    """
    assert(output.size() == label.size())

    labels = torch.ge(label, 0.5).float()

    num_labels_pos = torch.sum(labels)
    num_labels_neg = torch.sum(1.0 - labels)
    num_total = num_labels_pos + num_labels_neg

    output_gt_zero = torch.ge(output, 0).float()
    loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
        1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

    loss_pos_pix = -torch.mul(labels, loss_val)
    loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

    if void_pixels is not None:
        w_void = torch.le(void_pixels, 0.5).float()
        loss_pos_pix = torch.mul(w_void, loss_pos_pix)
        loss_neg_pix = torch.mul(w_void, loss_neg_pix)
        num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()

    loss_pos = torch.sum(loss_pos_pix)
    loss_neg = torch.sum(loss_neg_pix)

    final_loss = num_labels_neg / num_total * loss_pos + num_labels_pos / num_total * loss_neg

    if size_average:
        final_loss /= np.prod(label.size())
    elif batch_average:
        final_loss /= label.size()[0]

    return final_loss


def _iou(pred, target, size_average = True):

    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        Ior1 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand1
        IoU1 = Iand1/Ior1

        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b

class IOU_LOSS(torch.nn.Module):
    def __init__(self, size_average = True):
        super(IOU_LOSS, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return _iou(pred, target, self.size_average)


def _iou_void(pred, target, size_average = True, void_pixels=None):
    
    if void_pixels is None:
        void_pixels = torch.zeros_like(target)
    
    b = pred.shape[0]
    IoU = 0.0
    for i in range(0,b):
        #compute the IoU of the foreground
	#source IoU between prediction and ground truth
        #Iand0 = torch.sum(target[i,:,:,:]*pred[i,:,:,:])
        #Ior0 = torch.sum(target[i,:,:,:]) + torch.sum(pred[i,:,:,:])-Iand0
        #IoU0 = Iand0/Ior0
        #print('Iand0:{:.2f}, Ior0:{:.2f}, IoU0:{:.4f}'.format(Iand0, Ior0, IoU0))
         
        Iand1 = torch.sum(target[i,:,:,:]*pred[i,:,:,:]*(1-void_pixels[i,:,:,:]))
        Ior1 = torch.sum(target[i,:,:,:]*(1-void_pixels[i,:,:,:])) + torch.sum(pred[i,:,:,:]*(1-void_pixels[i,:,:,:]))-Iand1
        IoU1 = Iand1/Ior1
        #print('Iand1:{:.2f}, Ior1:{:.2f}, IoU1:{:.4f}'.format(Iand1, Ior1, IoU1))
        
        #IoU loss is (1-IoU1)
        IoU = IoU + (1-IoU1)

    return IoU/b


class IOU_LOSS_VOID(torch.nn.Module):
    def __init__(self):
        super(IOU_LOSS_VOID, self).__init__()

    def forward(self, pred, target, size_average=True, void_pixels=None):

        return _iou_void(pred, target, size_average=size_average, void_pixels=void_pixels)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM_LOSS(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM_LOSS, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


class LovaszSoftmax_LOSS(nn.Module):
    def __init__(self, reduction='mean'):
        super(LovaszSoftmax_LOSS, self).__init__()
        self.reduction = reduction

    def prob_flatten(self, input, target):
        assert input.dim() in [4, 5]
        num_class = input.size(1)
        if input.dim() == 4:
            input = input.permute(0, 2, 3, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        elif input.dim() == 5:
            input = input.permute(0, 2, 3, 4, 1).contiguous()
            input_flatten = input.view(-1, num_class)
        target_flatten = target.view(-1)
        return input_flatten, target_flatten

    def lovasz_softmax_flat(self, inputs, targets):
        num_classes = inputs.size(1)
        losses = []
        for c in range(num_classes):
            target_c = (targets == c).float()
            if num_classes == 1:
                input_c = inputs[:, 0]
            else:
                input_c = inputs[:, c]
            loss_c = (torch.autograd.Variable(target_c) - input_c).abs()
            loss_c_sorted, loss_index = torch.sort(loss_c, 0, descending=True)
            target_c_sorted = target_c[loss_index]
            losses.append(torch.dot(loss_c_sorted, torch.autograd.Variable(lovasz_grad(target_c_sorted))))
        losses = torch.stack(losses)

        if self.reduction == 'none':
            loss = losses
        elif self.reduction == 'sum':
            loss = losses.sum()
        else:
            loss = losses.mean()
        return loss

    def forward(self, inputs, targets):
        inputs, targets = self.prob_flatten(inputs, targets)
        losses = self.lovasz_softmax_flat(inputs, targets)
        return losses


def remove_prefix_state_dict(state_dict, prefix="module"):
    """
    remove prefix from the key of pretrained state dict for Data-Parallel
    """
    new_state_dict = {}
    first_state_name = list(state_dict.keys())[0]
    if not first_state_name.startswith(prefix):
        for key, value in state_dict.items():
            new_state_dict[key] = state_dict[key].float()
    else:
        for key, value in state_dict.items():
            new_state_dict[key[len(prefix)+1:]] = state_dict[key].float()
    return new_state_dict


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    import math
    if norm_type == math.inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    "[ref]--> https://medium.com/@ageitgey/the-dumb-reason-your-fancy-computer-vision-app-isnt-working-exif-orientation-73166c7d39da"
    # Load the image with PIL
    img = PIL.Image.open(file)

    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        img = PIL.ImageOps.exif_transpose(img)
    else:
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)

    return np.array(img)

Kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W 
    N, C, H, W = pred.shape

    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred<1.0/255.0] = 0
    uncertain_area[pred>1-1.0/255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n,0,:,:] # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, Kernels[width])
        uncertain_area[n,0,:,:] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).cuda()

    return weight


def get_masked_local_from_global(global_sigmoid, local_sigmoid):
	values, index = torch.max(global_sigmoid,1)
	index = index[:,None,:,:].float()
	### index <===> [0, 1, 2]
	### bg_mask <===> [1, 0, 0]
	bg_mask = index.clone()
	bg_mask[bg_mask==2]=1
	bg_mask = 1- bg_mask
	### trimap_mask <===> [0, 1, 0]
	trimap_mask = index.clone()
	trimap_mask[trimap_mask==2]=0
	### fg_mask <===> [0, 0, 1]
	fg_mask = index.clone()
	fg_mask[fg_mask==1]=0
	fg_mask[fg_mask==2]=1
	fusion_sigmoid = local_sigmoid*trimap_mask+fg_mask
	return fusion_sigmoid


class JSD(nn.Module):
    
    def __init__(self):
        super(JSD, self).__init__()
    
    def forward(self, net_1_logits, net_2_logits):
        net_1_probs =  F.softmax(net_1_logits, dim=1)
        net_2_probs=  F.softmax(net_2_logits, dim=1)

        m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), m, reduction="mean") 
        loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), m, reduction="mean") 
     
        return (0.5 * loss)


def consistency_loss(features, loss_func='mse'):
    """Computes the consistency loss between batches
        Input:
            features: B, C, H, W
        Return:
            consistency loss based on MSE / JS 
    """
    b = features.shape[0]
    if loss_func == 'mse':
        criterion = nn.MSELoss(reduction="mean")
    elif loss_func == 'js':
        criterion = JSD()
    loss = 0.0
    for i in range(b):
        for j in range(i+1, b):
            loss += criterion(features[i:i+1, :, :, :], features[j:j+1, :, :, :])
    loss /= b*(b-1) / 2.
    
    return loss
