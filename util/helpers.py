import os
import math, copy
import torch, cv2
import random
import numpy as np
from PIL import Image
from scipy.interpolate import UnivariateSpline
import warnings

def dilate(im, kernel=20):
    kernel = np.ones((kernel, kernel), np.uint8)
    print(len(np.where(im > 0.5)[0]), len(np.where(cv2.dilate(im, kernel) > 0.5)[0]))
    return cv2.dilate(im, kernel)

    
def tens2image(im):
    if im.size()[0] == 1:
        tmp = np.squeeze(im.numpy(), axis=0)
    else:
        tmp = im.numpy()
    if tmp.ndim == 2:
        return tmp
    else:
        return tmp.transpose((1, 2, 0)) # C, H, W -> H, W, C


def crop2fullmask(crop_mask, bbox, im=None, im_size=None, zero_pad=False, relax=0, mask_relax=True,
                  #interpolation=cv2.INTER_CUBIC, scikit=False):
                  interpolation=cv2.INTER_LINEAR, scikit=False):
    if scikit:
        from skimage.transform import resize as sk_resize
    assert(not(im is None and im_size is None)), 'You have to provide an image or the image size'
    if im is None:
        im_si = im_size
    else:
        im_si = im.shape
    # Borers of image
    bounds = (0, 0, im_si[1] - 1, im_si[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    # Bounding box of initial mask
    bbox_init = (bbox[0] + relax,
                 bbox[1] + relax,
                 bbox[2] - relax,
                 bbox[3] - relax)

    if zero_pad:
        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])
    else:
        assert((bbox == bbox_valid).all())
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

    if scikit:
        crop_mask = sk_resize(crop_mask, (bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), order=0, mode='constant').astype(crop_mask.dtype)
    else:
        crop_mask = cv2.resize(crop_mask, (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1), interpolation=interpolation)
    result_ = np.zeros(im_si)
    result_[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1] = \
        crop_mask[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1]

    result = np.zeros(im_si)
    if mask_relax:
        result[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1] = \
            result_[bbox_init[1]:bbox_init[3]+1, bbox_init[0]:bbox_init[2]+1]
    else:
        result = result_

    return result


def align2fullmask(crop_mask, im_size, points, relax=0):
    mask = np.zeros(im_size)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    
    pad = int(relax)
    if relax > 0 and relax < 1:
        x_len = x_max - x_min + 1
        y_len = y_max - y_min + 1
        pad = int(relax * max(x_len, y_len))
    # print('align original image', y_min, y_max+1, x_min, x_max+1)
    # print('aligh crop iamge', pad, pad+y_max-y_min+1, pad, pad+x_max-x_min+1)
    # import IPython
    # IPython.embed()
    mask[y_min:y_max + 1, x_min:x_max+1] = crop_mask[pad:pad+y_max-y_min+1, pad:pad+x_max-x_min+1]
    return mask


def overlay_mask(im, ma, colors=None, alpha=0.5):
    assert np.max(im) <= 1.0
    if colors is None:
        colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    else:
        colors = np.append([[0.,0.,0.]], colors, axis=0);

    if ma.ndim == 3:
        assert len(colors) >= ma.shape[0], 'Not enough colors'
    ma = ma.astype(np.bool)
    im = im.astype(np.float32)

    if ma.ndim == 2:
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[1, :3]   # np.array([0,0,255])/255.0
    else:
        fg = []
        for n in range(ma.ndim):
            fg.append(im * alpha + np.ones(im.shape) * (1 - alpha) * colors[1+n, :3])
    # Whiten background
    bg = im.copy()
    if ma.ndim == 2:
        bg[ma == 0] = im[ma == 0]
        bg[ma == 1] = fg[ma == 1]
        total_ma = ma
    else:
        total_ma = np.zeros([ma.shape[1], ma.shape[2]])
        for n in range(ma.shape[0]):
            tmp_ma = ma[n, :, :]
            total_ma = np.logical_or(tmp_ma, total_ma)
            tmp_fg = fg[n]
            bg[tmp_ma == 1] = tmp_fg[tmp_ma == 1]
        bg[total_ma == 0] = im[total_ma == 0]

    # [-2:] is s trick to be compatible both with opencv 2 and 3
    contours = cv2.findContours(total_ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    cv2.drawContours(bg, contours[0], -1, (0.0, 0.0, 0.0), 1)

    return bg

def overlay_masks(im, masks, alpha=0.5):
    colors = np.load(os.path.join(os.path.dirname(__file__), 'pascal_map.npy'))/255.
    
    if isinstance(masks, np.ndarray):
        masks = [masks]

    assert len(colors) >= len(masks), 'Not enough colors'

    ov = im.copy()
    im = im.astype(np.float32)
    total_ma = np.zeros([im.shape[0], im.shape[1]])
    i = 1
    for ma in masks:
        ma = ma.astype(np.bool)
        fg = im * alpha+np.ones(im.shape) * (1 - alpha) * colors[i, :3]   # np.array([0,0,255])/255.0
        i = i + 1
        ov[ma == 1] = fg[ma == 1]
        total_ma += ma

        # [-2:] is s trick to be compatible both with opencv 2 and 3
        contours = cv2.findContours(ma.copy().astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        cv2.drawContours(ov, contours[0], -1, (0.0, 0.0, 0.0), 1)
    ov[total_ma == 0] = im[total_ma == 0]

    return ov


def extreme_points(mask, pert):
    def find_point(id_x, id_y, ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return [id_x[sel_id], id_y[sel_id]]

    # List of coordinates of the mask
    # inds_y, inds_x = np.where(mask > 0.5)
    inds_y, inds_x = np.where(mask > 0)

    # Find extreme points
    return np.array([find_point(inds_x, inds_y, np.where(inds_x <= np.min(inds_x)+pert)), # left
                     find_point(inds_x, inds_y, np.where(inds_x >= np.max(inds_x)-pert)), # right
                     find_point(inds_x, inds_y, np.where(inds_y <= np.min(inds_y)+pert)), # top
                     find_point(inds_x, inds_y, np.where(inds_y >= np.max(inds_y)-pert)) # bottom
                     ])

from scipy import ndimage    
def getPositon(distance_transform):
    a = np.mat(distance_transform)
    
    raw, column = a.shape# get the matrix of a raw and column
    
    _positon = np.argmax(a)# get the index of max in the a

#    print _positon 
    m, n = divmod(_positon, column)
    raw=m
    column=n

    return  raw,column

def in_points(mask, pert):
     """Generating foreground points"""
     randombox = 30
     map_xor = (mask > 0)
     h,w = map_xor.shape
     map_xor_new = np.zeros((h+2,w+2))
     map_xor_new[1:(h+1),1:(w+1)] = map_xor[:,:]
     distance_transform=ndimage.distance_transform_edt(map_xor_new)
     distance_transform_back = distance_transform[1:(h+1),1:(w+1)] 
     
     raw,column=getPositon(distance_transform_back)
     center_point = [column,raw]

    #  center_mask = np.zeros(mask.shape)
    #  center_mask[max(raw-randombox,0):min(raw+randombox,h-1), max(column-randombox,0):min(column+randombox, w-1)]=1 
    #  center_pointmask = (center_mask>0.5) & (mask > 0.5) 
    #  inds_y2, inds_x2 = np.where(center_pointmask > 0.5)
    #  y_index = random.randint(0, len(inds_y2)-1)
    #  center_point = [inds_x2[y_index], inds_y2[y_index]]

     return np.array([center_point])

def out_points(mask, pert):
	"""Generating Background points"""
	# List of coordinates of the mask
	inds_y, inds_x = np.where(mask > 0)
	h, w = mask.shape
	pad_pixel = 10
	
	# Find bound (top,bottom,left,right)
	x1 = max(0,   np.min(inds_x) - pad_pixel)
	y1 = max(0,   np.min(inds_y) - pad_pixel)
	x2 = min(w-1, np.max(inds_x) + pad_pixel)
	y2 = min(h-1, np.max(inds_y) + pad_pixel)
	
	return np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])

def out_points_mask(mask, pert):
	"""Generating Background points"""
	# List of coordinates of the mask
	inds_y, inds_x = np.where(mask > 0)
	h, w = mask.shape
	pad_pixel = 10
	
	# Find bound (top,bottom,left,right)
	x1 = max(0,   np.min(inds_x) - pad_pixel)
	y1 = max(0,   np.min(inds_y) - pad_pixel)
	x2 = min(w-1, np.max(inds_x) + pad_pixel)
	y2 = min(h-1, np.max(inds_y) + pad_pixel)
	
	return np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])

def get_bbox(mask, points=None, pad=0, zero_pad=False, use_roimasking=False):
    if points is not None:
        inds = np.flip(points.transpose(), axis=0)
    else:
        inds = np.where(mask > 0)

    if inds[0].shape[0] == 0:
        return None

    if zero_pad:
        x_min_bound = -np.inf
        y_min_bound = -np.inf
        x_max_bound = np.inf
        y_max_bound = np.inf
    else:
        x_min_bound = 0
        y_min_bound = 0
        x_max_bound = mask.shape[1] - 1
        y_max_bound = mask.shape[0] - 1

    x_min = max(inds[1].min() - pad, x_min_bound)
    y_min = max(inds[0].min() - pad, y_min_bound)
    x_max = min(inds[1].max() + pad, x_max_bound)
    y_max = min(inds[0].max() + pad, y_max_bound)
    if not use_roimasking:
        return x_min, y_min, x_max, y_max
    else:
        # initial bbox without pad
        x_min_ini = max(inds[1].min(), x_min_bound)
        y_min_ini = max(inds[0].min(), y_min_bound)
        x_max_ini = min(inds[1].max(), x_max_bound)
        y_max_ini = min(inds[0].max(), y_max_bound)
        return x_min, y_min, x_max, y_max, x_min_ini, y_min_ini, x_max_ini, y_max_ini


def crop_from_bbox(img, bbox, zero_pad=False, use_roimasking=False):
    # Borders of image
    bounds = (0, 0, img.shape[1] - 1, img.shape[0] - 1)

    # Valid bounding box locations as (x_min, y_min, x_max, y_max)
    bbox_valid = (max(bbox[0], bounds[0]),
                  max(bbox[1], bounds[1]),
                  min(bbox[2], bounds[2]),
                  min(bbox[3], bounds[3]))

    if zero_pad:
        # Initialize crop size (first 2 dimensions)
        crop = np.zeros((bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1), dtype=img.dtype)

        # Offsets for x and y
        offsets = (-bbox[0], -bbox[1])

    else:
        assert(bbox == bbox_valid)
        crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=img.dtype)
        offsets = (-bbox_valid[0], -bbox_valid[1])

    # Simple per element addition in the tuple
    inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))
    if use_roimasking:
        bbox_valid_nopad = (max(bbox[4], bounds[0]),
                            max(bbox[5], bounds[1]),
                            min(bbox[6], bounds[2]),
                            min(bbox[7], bounds[3]))
        inds_nopad = tuple(map(sum, zip(bbox_valid_nopad, offsets + offsets)))

    img = np.squeeze(img)
    
    if not use_roimasking:
        if img.ndim == 2:
            crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
                img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
        else:
            crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
            crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
                img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]
    else:
        if img.ndim == 2:
            crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
                - img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]

            crop[inds_nopad[1]:inds_nopad[3] + 1, inds_nopad[0]:inds_nopad[2] + 1] = \
                img[bbox_valid_nopad[1]:bbox_valid_nopad[3] + 1, bbox_valid_nopad[0]:bbox_valid_nopad[2] + 1]
        else:
            crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
        
            crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
                - img[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]

            crop[inds_nopad[1]:inds_nopad[3] + 1, inds_nopad[0]:inds_nopad[2] + 1] = \
                img[bbox_valid_nopad[1]:bbox_valid_nopad[3] + 1, bbox_valid_nopad[0]:bbox_valid_nopad[2] + 1]

    return crop


def fixed_resize(sample, resolution, flagval=None):

    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_LINEAR  #cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[np.argmax(sample.shape[:2])] = int(round(float(resolution)/np.min(sample.shape[:2])*np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def crop_from_mask(img, mask, relax=0, zero_pad=False, use_roimasking = False):
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, dsize=tuple(reversed(img.shape[:2])), interpolation=cv2.INTER_NEAREST)

    assert(mask.shape[:2] == img.shape[:2])

    bbox = get_bbox(mask, pad=relax, zero_pad=zero_pad, use_roimasking = use_roimasking)

    if bbox is None:
        return None

    crop = crop_from_bbox(img, bbox, zero_pad, use_roimasking = use_roimasking)

    return crop


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def make_gt(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=img.dtype)

    return gt

def make_gt_bbox(img, labels, sigma=10, one_mask_per_point=False):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    h, w = img.shape[:2]
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)
        if labels.ndim == 1:
            labels = labels[np.newaxis]
        if one_mask_per_point:
            gt = np.zeros(shape=(h, w, labels.shape[0]))
            for ii in range(labels.shape[0]):
                gt[:, :, ii] = make_gaussian((h, w), center=labels[ii, :], sigma=sigma)
        else:
            gt = np.zeros(shape=(h, w), dtype=np.float64)
            for ii in range(labels.shape[0]):
                gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))
                x1, y1 = labels[0]
                x2, y2 = labels[3]
                gt[y1:y2, x1:x2] = 255

    gt = gt.astype(dtype=img.dtype)

    return gt

def cstm_normalize(im, max_value):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value*(im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key+':'+str(val)+'\n')
    log_file.close()

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def save_mask(results, mask_path):
    mask = np.zeros(results[0].shape)
    for ii, r in enumerate(results):
        mask[r] = ii + 1
    result = Image.fromarray(mask.astype(np.uint8))
    result.putpalette(list(color_map(80).flatten()))
    result.save(mask_path)


def B_spline(control_points, num_i, s=0.5):
    '''
    Using B_spline to interpolate
    args
        control_points: list of control_points
        num_i: number of interpolation points(between two control points or end points)
        s: hyper parameter for b-spline
    return
        points: list of interpolated points
    '''
    points = []
    num_c = len(control_points)
    for i in range(num_c):
        for t in range(num_i):
            i0 = max(0, i - 1)
            i1 = i
            i2 = min(num_c - 1, i + 1)
            i3 = min(num_c - 1, i + 2)
            f = t * 1.0 / num_i
            c0 = (1.0 / 3 - s) * (f ** 3) + (2 * s - 1.0 / 2) * (f ** 2) - s * (f) + 1.0 / 6
            c1 = (1 - s) * (f ** 3) + (s - 3.0 / 2) * (f ** 2) + 2.0 / 3
            c2 = (s - 1) * (f ** 3) + (3.0 / 2 - 2 * s) * (f ** 2) + s * (f) + 1.0 / 6
            c3 = (s - 1.0 / 3) * (f ** 3) + (1.0 / 2 - s) * (f ** 2)
            tmp_point = control_points[i0] * c0 + control_points[i1] * c1 + \
                        control_points[i2] * c2 + control_points[i3] * c3
            points.append(tmp_point.astype(np.int))
    return points


def generate_scribble_strictly(mask, num_c=3, num_i=50, coverage_area=0.1, width=10, best_out_of=5):
    '''
    generate one B-spline with 2 end points and several control points to be a scribble
    args 
        mask: 2D np.array shape: H x W dtype bool(1 for target mask, 0 for others)
        num_c: number of control points (points except for the two end points)
        num_i: number of interpolation points(between two control points or end points)
    return 
        scribble points: 2D np.array shape:  L(number of points) x 2 (0 for x, 1 for y)
    '''
    H, W = mask.shape
    mask_points = np.where(mask > 0)
    mask_points = np.array([mask_points[1], mask_points[0]])
    num_mask_points = mask_points.shape[1]
    total_area = mask.sum()
    max_coverage = 0
    best_scribbles = []
    num_of_candidates = 0
    number_of_out_of_bound = 0
    while (num_of_candidates < best_out_of):
        scribble_points = []
        for i in range(num_c):
            sample_index = int(np.random.rand() * num_mask_points)
            control_points = mask_points[:, sample_index]
            scribble_points.append(control_points)
        scribble_points = B_spline(scribble_points, num_i)

        # check out_of_bound_point
        new_scribble_points = []
        out_of_bound = False
        for i in range(len(scribble_points)):
            if mask[scribble_points[i][1], scribble_points[i][0]] < 1 and number_of_out_of_bound < 20:
                out_of_bound = True
                break
            else:
                new_scribble_points.append(scribble_points[i])
        if out_of_bound:
            number_of_out_of_bound += 1
            continue
        number_of_out_of_bound = 0

        # remove duplicate points
        num_of_candidates += 1
        scribble_points = np.array(new_scribble_points)
        # scribble_points = np.unique(scribble_points, axis=0)

        remain_mask = mask.copy()
        for i in range(len(scribble_points)):
            x = scribble_points[i, 0]
            y = scribble_points[i, 1]
            t = max(0, y - width)
            b = min(H - 1, y + width)
            l = max(0, x - width)
            r = min(W - 1, x + width)
            remain_mask[t:b, l:r] = 0
        remain_area = remain_mask.sum()
        if (1 - remain_area * 1.0 / total_area) > max_coverage:
            max_coverage = (1 - remain_area * 1.0 / total_area)
            best_scribbles = scribble_points
    return best_scribbles


def generate_trimap_with_gaussian(mask):
    mask = np.array(mask,np.uint8)
    h, w = mask.shape[:2]
    if np.max(mask) == 1:
        mask = mask * 255
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    trimap = np.zeros((h, w), np.uint8)
    trimap[mask == 255] = 255
    # image, contours, hierarchy = cv2.findContours(mask, 2, 1)
    contours, hierarchy = cv2.findContours(mask, 2, 1)
    # print(np.array(contours).shape)
    para_area = np.sqrt(len(np.where(mask > 0)[0]))
    thickness = max(int(para_area*0.03), 2)
    for idx, cnt in enumerate(contours):
        arcLength = cv2.arcLength(cnt, True)
        para_step = max(int(arcLength/300), 1)
        cnt = np.squeeze(cnt, axis=1)
        link_flag = True
        lines = []
        line = []
        for point in cnt:
            x, y = point
            if math.fabs(x) < 2 or math.fabs(x - w) < 2 or math.fabs(y - 0) < 2 or math.fabs(y - h) < 2:
                link_flag=False
                if len(line) != 0:
                    # print(line)
                    lines.append(copy.deepcopy(line))
                line = []
                continue
            else:
                line.append([x, y])
                # print(line)
        if len(line) != 0:
            lines.append(copy.deepcopy(line))

        for line in lines:
            # print(line)
            line_sample = line[0:len(line):para_step]
            if line[-1] not in line_sample:
                line_sample.append(line[-1])
            line_sample = np.array(line_sample)
            para_rand = para_area / 50
            rand_mask = np.random.standard_normal(line_sample.shape) / 5.0
            line_sample = line_sample + rand_mask
            line_sample = np.array(line_sample, np.int32)
            # print(line)
            cv2.polylines(trimap, [line_sample], link_flag, 128, thickness=thickness, lineType=cv2.LINE_AA)
    return trimap


def clamp(input, min=None, max=None):
    if min is not None and input < min:
        return min
    elif max is not None and input > max:
        return max
    else:
        return input


def produce_trimap(mask):
    """
    mask -> (erosion & dilation) -> extract contour -> extract lines -> random shift -> b-spline fitting -> draw lines
    :param mask:
    :return: trimap
    """
    mask = np.array(mask,np.uint8)
    h, w = mask.shape[:2]

    if np.max(mask) == 1:
        mask = mask * 255
    else:
        mask = mask

    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    trimap = np.zeros((h, w), np.uint8)
    trimap[mask == 255] = 255

    mask_area = len(np.where(mask > 0)[0])
    mask_linear = np.sqrt(mask_area)

    for m in [mask]: # or [eroded, dilated]
        # image, contours, hierarchy = cv2.findContours(m, 2, 1)
        contours, hierarchy = cv2.findContours(mask, 2, 1)

        """ (para) thickness """
        thickness = clamp(int(mask_linear * 0.04), min=14, max=23)

        for idx, cnt in enumerate(contours):
            cnt = np.squeeze(cnt, axis=1)
            link_flag = True
            lines = []
            line = []

            """ === split contours into lines (because of boundary) === """
            for point in cnt:
                x, y = point
                if math.fabs(x) < 2 or math.fabs(x - w) < 2 or math.fabs(y - 0) < 2 or math.fabs(y - h) < 2:
                    link_flag = False
                    if len(line) != 0:
                        lines.append(copy.deepcopy(line))
                    line = []
                    continue
                else:
                    line.append([x, y])

            if len(line) != 0:
                lines.append(copy.deepcopy(line))

            for line in lines:
                """ (para) sample_step """
                arc_length = cv2.arcLength(cnt, link_flag)
                sample_step = clamp(int(arc_length / 230), min=1)
                if len(line)/sample_step > 200:
                    sample_step = int(len(line) / 200) + 1
                line_sample = line[0:len(line):sample_step]

                if len(line_sample) > 220:
                    assert RuntimeError("the sampled point should be no more than 200")

                if line[-1] not in line_sample:
                    line_sample.append(line[-1])
                line_sample = np.array(line_sample)

                """ (para) rand_shift """
                rand_shift = clamp(arc_length / 2700, min=0)
                print(rand_shift)
                rand_mask = np.random.normal(0, rand_shift, line_sample.shape)
                line_sample = line_sample + rand_mask
                line_sample = np.array(line_sample, np.int32)

                """ === use b-spline to interpolate the curve from the sampled points === """
                if len(line_sample) > 3:
                    distance = np.cumsum(np.sqrt(np.sum(np.diff(line_sample, axis=0) ** 2, axis=1)))
                    print(distance.shape)
                    distance = np.insert(distance, 0, 0) / distance[-1]

                    # Build a list of the spline function, one for each dimension:
                    splines = [UnivariateSpline(distance, coords, k=3, s=150) for coords in line_sample.T]

                    # Computed the spline for the asked distances:
                    alpha = np.linspace(0, 1, 150)
                    line_inter = np.vstack([spl(alpha) for spl in splines]).T
                else:
                    line_inter = line_sample

                line_inter = np.array(line_inter, np.int32)
                cv2.polylines(trimap, [line_inter], link_flag, 128, thickness=thickness, lineType=cv2.LINE_AA)

    return trimap


def unified_trimap_transform(trimap, sample_name, split_dir):
    """
    Transform standard trimap to unified type (Deep Automatic Image Matting, IJCAI 2021)
    """
    with open(os.path.join(os.path.join(split_dir, 'train_so.txt')), "r") as f:
        so_list = f.read().splitlines()
    with open(os.path.join(os.path.join(split_dir, 'train_stm.txt')), "r") as f:
        stm_list = f.read().splitlines()
    with open(os.path.join(os.path.join(split_dir, 'train_nstm.txt')), "r") as f:
        nstm_list = f.read().splitlines()
    
    sample_name = '_'.join(sample_name.split('_')[:-1])
    
    if sample_name in so_list:
        trimap_unified = trimap
    elif sample_name in stm_list:
        trimap[trimap == 2] = 1
        trimap_unified = trimap
    elif sample_name in nstm_list:
        trimap_unified = np.ones_like(trimap)
    else:
        warnings.warn("%s not in all of the data type list"%(sample_name))
        trimap_unified = trimap
    
    return trimap_unified