import torch, cv2
import numpy.random as random
import numpy as np
import util.helpers as helpers
from util.helpers import *
from torchvision import transforms
import copy
from skimage import draw
from scipy import ndimage
import logging
import numbers
import torch.nn.functional as F


# ======== Data Transformation Tools ========
interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]
def maybe_random_interp(cv2_interp, random_interp=False):
    if random_interp:
        return np.random.choice(interp_list)
    else:
        return cv2_interp


class ScaleNRotate(object):
    """Scale (zoom-in, zoom-out) and Rotate the image and the ground truth.
    Args:
        two possibilities:
        1.  rots (tuple): (minimum, maximum) rotation angle
            scales (tuple): (minimum, maximum) scale
        2.  rots [list]: list of fixed possible rotation angles
            scales [list]: list of fixed possible scales
    """
    def __init__(self, rots=(-30, 30), scales=(.75, 1.25), semseg=True):
        assert (isinstance(rots, type(scales)))
        self.rots = rots
        self.scales = scales
        self.semseg = semseg

    def __call__(self, sample):

        if type(self.rots) == tuple:
            # Continuous range of scales and rotations
            rot = (self.rots[1] - self.rots[0]) * random.random() - \
                  (self.rots[1] - self.rots[0])/2

            sc = (self.scales[1] - self.scales[0]) * random.random() - \
                 (self.scales[1] - self.scales[0]) / 2 + 1
        elif type(self.rots) == list:
            # Fixed range of scales and rotations
            rot = self.rots[random.randint(0, len(self.rots))]
            sc = self.scales[random.randint(0, len(self.scales))]
        
        for elem in sample.keys():
            if 'meta' in elem or sample[elem] is None:
                continue

            tmp = sample[elem]
            h, w = tmp.shape[:2]
            center = (w / 2, h / 2)

            assert(center != 0)  # Strange behaviour warpAffine
            M = cv2.getRotationMatrix2D(center, rot, sc)

            if ((tmp == 0) | (tmp == 1)).all():
                flagval = cv2.INTER_NEAREST
            elif 'gt' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            elif 'scribble' in elem and self.semseg:
                flagval = cv2.INTER_NEAREST
            else:
                flagval = cv2.INTER_LINEAR #cv2.INTER_CUBIC
            tmp = cv2.warpAffine(tmp, M, (w, h), flags=flagval)

            sample[elem] = tmp
        return sample

    def __str__(self):
        return 'ScaleNRotate:(rot='+str(self.rots)+',scale='+str(self.scales)+')'


class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        resolutions (dict): the list of resolutions
    """
    def __init__(self, resolutions=None, flagvals=None, del_elem = True):
        self.resolutions = resolutions
        self.flagvals = flagvals
        self.del_elem = del_elem
        if self.flagvals is not None:
            assert(len(self.resolutions) == len(self.flagvals))

    def __call__(self, sample):

        # Fixed range of scales
        if self.resolutions is None:
            return sample

        elems = list(sample.keys())

        for elem in elems:
            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if 'extreme_points_coord' in elem and elem in self.resolutions:
                bbox = sample['bbox']
                crop_size = np.array([bbox[3]-bbox[1]+1, bbox[4]-bbox[2]+1])
                res = np.array(self.resolutions[elem]).astype(np.float32)
                sample[elem] = np.round(sample[elem]*res/crop_size).astype(np.int)
                continue
            if elem in self.resolutions:
                if self.resolutions[elem] is None:
                    continue
                if isinstance(sample[elem], list):
                    if sample[elem][0].ndim == 3:
                        output_size = np.append(self.resolutions[elem], [3, len(sample[elem])])
                    else:
                        output_size = np.append(self.resolutions[elem], len(sample[elem]))
                    tmp = sample[elem]
                    sample[elem] = np.zeros(output_size, dtype=np.float32)
                    for ii, crop in enumerate(tmp):
                        if self.flagvals is None:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem])
                        else:
                            sample[elem][..., ii] = helpers.fixed_resize(crop, self.resolutions[elem], flagval=self.flagvals[elem])
                else:
                    if self.flagvals is None:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem])
                    else:
                        sample[elem] = helpers.fixed_resize(sample[elem], self.resolutions[elem], flagval=self.flagvals[elem])
            else:
                if self.del_elem:
                    del sample[elem] # must del when training

        return sample

    def __str__(self):
        return 'FixedResize:'+str(self.resolutions)


class RandomCrop(object):
    """
    Crop randomly the image in a sample, retain the center 1/4 images, and resize to 'output_size'

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, crop_elems=('fg', 'alpha', 'bg', 'trimap'), output_size=(512, 512)):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.margin = output_size[0] // 2
        self.logger = logging.getLogger("Logger")

    def __call__(self, sample):
        fg, alpha, trimap, name = sample['fg'], sample['alpha'], sample['trimap'], sample['meta']['image']
        bg = sample['bg']
        h, w = trimap.shape
        bg = cv2.resize(bg, (w, h), interpolation=maybe_random_interp(cv2.INTER_CUBIC))
        if w < self.output_size[0] + 1 or h < self.output_size[1] + 1:
            ratio = 1.1 * self.output_size[0] / h if h < w else 1.1 * self.output_size[1] / w
            # self.logger.warning("Size of {} is {}.".format(name, (h, w)))
            while h < self.output_size[0] + 1 or w < self.output_size[1] + 1:
                fg = cv2.resize(fg, (int(w * ratio), int(h * ratio)),
                                interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                alpha = cv2.resize(alpha, (int(w * ratio), int(h * ratio)),
                                   interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                trimap = cv2.resize(trimap, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_NEAREST)
                bg = cv2.resize(bg, (int(w * ratio), int(h * ratio)),
                                interpolation=maybe_random_interp(cv2.INTER_CUBIC))
                h, w = trimap.shape
        small_trimap = cv2.resize(trimap, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
        unknown_list = list(zip(*np.where(small_trimap[self.margin // 4:(h - self.margin) // 4,
                                          self.margin // 4:(w - self.margin) // 4] == 128)))
        unknown_num = len(unknown_list)
        if len(unknown_list) < 10:
            # self.logger.warning("{} does not have enough unknown area for crop.".format(name))
            left_top = (
            np.random.randint(0, h - self.output_size[0] + 1), np.random.randint(0, w - self.output_size[1] + 1))
        else:
            idx = np.random.randint(unknown_num)
            left_top = (unknown_list[idx][0] * 4, unknown_list[idx][1] * 4)

        fg_crop = fg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1], :]
        alpha_crop = alpha[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1]]
        bg_crop = bg[left_top[0]:left_top[0] + self.output_size[0], left_top[1]:left_top[1] + self.output_size[1], :]
        trimap_crop = trimap[left_top[0]:left_top[0] + self.output_size[0],
                      left_top[1]:left_top[1] + self.output_size[1]]

        if len(np.where(trimap == 128)[0]) == 0:
            self.logger.error("{} does not have enough unknown area for crop. Resized to target size."
                              "left_top: {}".format(name, left_top))
            fg_crop = cv2.resize(fg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            alpha_crop = cv2.resize(alpha, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_NEAREST))
            trimap_crop = cv2.resize(trimap, self.output_size[::-1], interpolation=cv2.INTER_NEAREST)
            bg_crop = cv2.resize(bg, self.output_size[::-1], interpolation=maybe_random_interp(cv2.INTER_CUBIC))
            # cv2.imwrite('../tmp/tmp.jpg', fg.astype(np.uint8))
            # cv2.imwrite('../tmp/tmp.png', (alpha*255).astype(np.uint8))
            # cv2.imwrite('../tmp/tmp2.png', trimap.astype(np.uint8))
            # raise ValueError("{} does    not have enough unknown area for crop.".format(name))

        sample['fg'], sample['alpha'], sample['trimap'] = fg_crop, alpha_crop, trimap_crop
        sample['bg'] = bg_crop

        return sample


class RandomAffine(object):
    """
    Random affine translation
    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, flip=None, resample=False, fillcolor=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor
        self.flip = flip

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, flip, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])
        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = (random.uniform(scale_ranges[0], scale_ranges[1]),
                     random.uniform(scale_ranges[0], scale_ranges[1]))
        else:
            scale = (1.0, 1.0)

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        if flip is not None:
            flip = (np.random.rand(2) < flip).astype(np.int) * 2 - 1

        return angle, translations, scale, shear, flip

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        rows, cols, ch = fg.shape
        if np.maximum(rows, cols) < 1024:
            params = self.get_params((0, 0), self.translate, self.scale, self.shear, self.flip, fg.size)
        else:
            params = self.get_params(self.degrees, self.translate, self.scale, self.shear, self.flip, fg.size)

        center = (cols * 0.5 + 0.5, rows * 0.5 + 0.5)
        M = self._get_inverse_affine_matrix(center, *params)
        M = np.array(M).reshape((2, 3))

        fg = cv2.warpAffine(fg, M, (cols, rows),
                            flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)
        alpha = cv2.warpAffine(alpha, M, (cols, rows),
                               flags=maybe_random_interp(cv2.INTER_NEAREST) + cv2.WARP_INVERSE_MAP)

        sample['fg'], sample['alpha'] = fg, alpha

        return sample

    @staticmethod
    def _get_inverse_affine_matrix(center, angle, translate, scale, shear, flip):
        # Helper method to compute inverse matrix for affine transformation

        # As it is explained in PIL.Image.rotate
        # We need compute INVERSE of affine transformation matrix: M = T * C * RSS * C^-1
        # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
        # C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
        # RSS is rotation with scale and shear matrix
        # It is different from the original function in torchvision
        # The order are changed to flip -> scale -> rotation -> shear
        # x and y have different scale factors
        # RSS(shear, a, scale, f) = [ cos(a + shear)*scale_x*f -sin(a + shear)*scale_y     0]
        # [ sin(a)*scale_x*f          cos(a)*scale_y             0]
        # [     0                       0                      1]
        # Thus, the inverse is M^-1 = C * RSS^-1 * C^-1 * T^-1

        angle = math.radians(angle)
        shear = math.radians(shear)
        scale_x = 1.0 / scale[0] * flip[0]
        scale_y = 1.0 / scale[1] * flip[1]

        # Inverted rotation matrix with scale and shear
        d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
        matrix = [
            math.cos(angle) * scale_x, math.sin(angle + shear) * scale_x, 0,
            -math.sin(angle) * scale_y, math.cos(angle + shear) * scale_y, 0
        ]
        matrix = [m / d for m in matrix]

        # Apply inverse of translation and of center translation: RSS^-1 * C^-1 * T^-1
        matrix[2] += matrix[0] * (-center[0] - translate[0]) + matrix[1] * (-center[1] - translate[1])
        matrix[5] += matrix[3] * (-center[0] - translate[0]) + matrix[4] * (-center[1] - translate[1])

        # Apply center translation: C * RSS^-1 * C^-1 * T^-1
        matrix[2] += center[0]
        matrix[5] += center[1]

        return matrix


class RandomJitter(object):
    """
    Random change the hue of the image
    """

    def __call__(self, sample):
        fg, alpha = sample['fg'], sample['alpha']
        # if alpha is all 0 skip
        if np.all(alpha == 0):
            return sample
        # convert to HSV space, convert to float32 image to keep precision during space conversion.
        fg = cv2.cvtColor(fg.astype(np.float32) / 255.0, cv2.COLOR_BGR2HSV)
        # Hue noise
        hue_jitter = np.random.randint(-40, 40)
        fg[:, :, 0] = np.remainder(fg[:, :, 0].astype(np.float32) + hue_jitter, 360)
        # Saturation noise
        sat_bar = fg[:, :, 1][alpha > 0].mean()
        sat_jitter = np.random.rand() * (1.1 - sat_bar) / 5 - (1.1 - sat_bar) / 10
        sat = fg[:, :, 1]
        sat = np.abs(sat + sat_jitter)
        sat[sat > 1] = 2 - sat[sat > 1]
        fg[:, :, 1] = sat
        # Value noise
        val_bar = fg[:, :, 2][alpha > 0].mean()
        val_jitter = np.random.rand() * (1.1 - val_bar) / 5 - (1.1 - val_bar) / 10
        val = fg[:, :, 2]
        val = np.abs(val + val_jitter)
        val[val > 1] = 2 - val[val > 1]
        fg[:, :, 2] = val
        # convert back to BGR space
        fg = cv2.cvtColor(fg, cv2.COLOR_HSV2BGR)
        sample['fg'] = fg * 255
 
        return sample


class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                if 'meta' in elem:
                    continue
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class Composite(object):
    def __call__(self, sample):
        fg, bg, alpha = sample['fg'], sample['bg'], sample['alpha']
        alpha[alpha < 0] = 0
        alpha[alpha > 1] = 1
        fg[fg < 0] = 0
        fg[fg > 255] = 255
        bg[bg < 0] = 0
        bg[bg > 255] = 255

        image = fg * alpha[:, :, None] + bg * (1 - alpha[:, :, None])
        sample['image'] = image
        return sample


class ExtremePoints(object):
    """
    Returns the four extreme points (left, right, top, bottom) (with some random perturbation) in a given binary mask
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, sigma=10, pert=0, elem='gt', p_type='', relax=0.3):
        self.sigma = sigma
        self.elem = elem
        self.p_type = p_type
        self.relax = relax
        if self.p_type == 'b_points':
            self.pert = 0
        else:
            self.pert = pert
        # self.scribble_elem = scribble_elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('ExtremePoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            if self.p_type == 'b_points':
                sample['box_points'] = np.zeros(_target.shape, dtype=_target.dtype)
            else:
                sample['extreme_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.extreme_points(_target, self.pert)

            if self.p_type == 'b_points':
                x_min = _points[0][0]
                x_max = _points[1][0]
                y_min = _points[2][1]
                y_max = _points[3][1]

                h, w = _target.shape

                crop_image = sample['crop_image']
                crop_image_x = crop_image[:,:,0].sum(0)
                crop_image_y = crop_image[:,:,0].sum(1)

                xx = np.where(crop_image_x!=0)[0]
                yy = np.where(crop_image_y!=0)[0]
                x_start = xx.min()
                x_end = xx.max()
                y_start = yy.min()
                y_end = yy.max()

                #_target_area
                # ([y1,y2,...],[x1,x2,...])
                _target_y, _target_x = np.where(_target > 0)

                padding_x = _target_x.min()//2
                padding_y = _target_y.min()//2

                x_min = max(x_start, x_min - padding_x)
                y_min = max(y_start, y_min - padding_y)
                x_max = min(x_end, x_max + padding_x)
                y_max = min(y_end, y_max + padding_y)
                _box_points = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_min], [x_max, y_max]])
                sample['box_points'] = helpers.make_gt(_target, _box_points, sigma=self.sigma, one_mask_per_point=False)

            else:
                sample['extreme_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)

        return sample

    def __str__(self):
        return 'ExtremePoints:(sigma='+str(self.sigma)+', pert='+str(self.pert)+', elem='+str(self.elem)+')'

class IOGPoints(object):
    """
    Returns the four background points (top-left, top-right, bottom-left, bottom-right) (with some random perturbation) in a given binary mask, and one foreground point inside the object.
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    p_type: `in` for inside point, `out` for outside points
    """
    def __init__(self, sigma=10, pert=0, elem='gt', p_type='out'):
        self.sigma = sigma
        self.pert = pert
        self.elem = elem
        self.p_type = p_type

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('IOGPoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        # print(sample['meta']['image'], _target.min(),_target.max())
        if np.max(_target) == 0:
            sample[self.p_type+'_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            if self.p_type=='in':
                _points = helpers.in_points(_target, self.pert)
            else:
                _points = helpers.out_points(_target, self.pert)

            sample[self.p_type+'_points'] = helpers.make_gt(_target, _points, sigma=self.sigma, one_mask_per_point=False)
        return sample

    def __str__(self):
        return 'IOGPoints:(sigma='+str(self.sigma)+', pert='+str(self.pert)+', elem='+str(self.elem)+', p_type='+str(self.p_type)+')'

class OutPoints(object):
    """
    Returns the four background points (top-left, top-right, bottom-left, bottom-right) (with some random perturbation) in a given binary mask, and one foreground point inside the object.
    sigma: sigma of Gaussian to create a heatmap from a point
    pert: number of pixels fo the maximum perturbation
    elem: which element of the sample to choose as the binary mask
    p_type: `in` for inside point, `out` for outside points
    """
    def __init__(self, sigma=10, pert=0, elem='gt'):
        self.sigma = sigma
        self.pert = pert
        self.elem = elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('IOGPoints not implemented for multiple object per image.')
        _target = sample[self.elem]
        # print(sample['meta']['image'], _target.min(),_target.max())
        if np.max(_target) == 0:
            sample['out_points'] = np.zeros(_target.shape, dtype=_target.dtype) #  TODO: handle one_mask_per_point case
        else:
            _points = helpers.out_points(_target, self.pert)

            sample['out_points'] = helpers.make_gt_bbox(_target, _points, sigma=self.sigma, one_mask_per_point=False)
        return sample

    def __str__(self):
        return 'Out Points:(sigma='+str(self.sigma)+', pert='+str(self.pert)+', elem='+str(self.elem) + ')'

class CentroidMap(object):
    """ Generate the centroid map of ground-truth
    """
    def __init__(self, elem='crop_gt', type='foreground'):
        self.elem = elem
        self.type = type
    
    def __call__(self, sample):
        gt = sample[self.elem]
        if self.type == 'foreground':
            foreground_dis = ndimage.morphology.distance_transform_edt(gt)
            sample['fore_centroid_map'] = foreground_dis.astype(gt.dtype)
        else:
            background_dis = ndimage.morphology.distance_transform_edt(1-gt)
            sample['back_centroid_map'] = background_dis.astype(gt.dtype)
        return sample


class VoidBound(object):
    """
    Fill semantic boundary with void pixels
    """
    def __init__(self, elem='crop_gt'):
        self.elem = elem

    def __call__(self, sample):
        crop_gt = sample[self.elem]
        crop_void = sample['crop_void_pixels']
        cv_gt = np.asarray(crop_gt*255, dtype=np.uint8)
        cv_void = np.asarray(crop_void*255, dtype=np.uint8)
        ret, pred_bin = cv2.threshold(cv_gt, 127, 255, cv2.THRESH_BINARY) 

        if cv2.__version__[0] == '3':
            image, contours, hierarchy = cv2.findContours(pred_bin, 3, 2)
        else:
            contours, hierarchy = cv2.findContours(pred_bin, 3, 2)
        
        cv_h, cv_w = cv_gt.shape
        cv_void2 = np.zeros((cv_h, cv_w, 3), dtype=np.uint8)
        cv_void4 = np.zeros((cv_h, cv_w, 3), dtype=np.uint8)
        cv_void8 = np.zeros((cv_h, cv_w, 3), dtype=np.uint8)

        cv2.drawContours(cv_void2, contours, -1, (255,255,255), 2) 
        cv2.drawContours(cv_void4, contours, -1, (255,255,255), 4) 
        cv2.drawContours(cv_void8, contours, -1, (255,255,255), 8) 

        void2 = (cv_void2[:,:,0]==255).astype(np.float32)
        void4 = (cv_void4[:,:,0]==255).astype(np.float32)
        void8 = (cv_void8[:,:,0]==255).astype(np.float32)
        sample['void2'] = void2
        sample['void4'] = void4
        sample['void8'] = void8
        return sample

class GenerateScribble(object):
    """
    Returns the scribble generated according to gt
    dilate: dilate kernel size
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, kernel=5, elem='gt'):
        self.kernel = kernel
        self.elem = elem

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('GenerateScribble not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            sample['scribble'] = np.zeros(_target.shape, dtype=_target.dtype)
        else:
            scribble_points = helpers.generate_scribble_strictly(_target, num_c=random.randint(3, 4))
            scribble_map = np.zeros(_target.shape).astype(np.float32)
            
            for point in scribble_points:
                scribble_map[point[1],point[0]] = 1
            '''
            # draw line with in the connect scribble points
            length = len(scribble_points)
            for ii in range(length - 1):
                x1, y1 = scribble_points[ii][0], scribble_points[ii][1]
                x2, y2 = scribble_points[ii+1][0], scribble_points[ii+1][1]
                xx, yy =draw.line(np.max(np.round(x1) - 1, 0), np.max(np.round(y1) - 1, 0), \
                                  np.max(np.round(x2) - 1, 0), np.max(np.round(y2) - 1, 0))
                scribble_map[yy, xx] = 1
            '''
            if self.kernel > 1:
                kernel_size = self.kernel
            else:
                bbox = sample['meta']['boundary']
                length_short_side = min(bbox[0] - bbox[1], bbox[2] - bbox[3])
                kernel_size = self.kernel * length_short_side
            dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
            sample['scribble'] = cv2.dilate(scribble_map, dilate_kernel)
        return sample

    def __str__(self):
        return 'GenerateScribble:(kernel='+str(self.kernel)+', elem='+str(self.elem)+')'


# GCA-Matting trimap generation
class GenTrimap(object):
    def __init__(self):
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]

    def __call__(self, sample):
        alpha = sample['alpha']
        # Adobe 1K
        fg_width = np.random.randint(1, 30)
        bg_width = np.random.randint(1, 30)
        fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
        bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
        fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
        bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

        trimap = np.ones_like(alpha) * 128
        trimap[fg_mask == 1] = 255
        trimap[bg_mask == 1] = 0

        sample['trimap'] = trimap
        return sample


# Generate binary mask from alpha
class GenMask(object):
    def __init__(self, mask_type='alpha', mask_thresh=0):
        self.mask_thresh = mask_thresh
        self.mask_type = mask_type

    def __call__(self, sample):
        alpha = sample['alpha']
        trimap = sample['trimap']
        if self.mask_type == 'alpha':    
            mask = (alpha > self.mask_thresh).astype(np.float32)
        elif self.mask_type == 'fg':
            mask = (trimap == 255).astype(np.float32)
        else:
            raise NotImplementedError('Wrong mask type.')
        sample['mask'] = mask
        return sample


# MattingSeg trimap generation 
class GenerateTrimap(object):
    """
    Returns the trimap generated according to gt
    dilate: dilate kernel size
    elem: which element of the sample to choose as the binary mask
    """
    def __init__(self, kernel=5, elem='gt', by_percent=False):
        self.kernel = kernel
        self.elem = elem
        self.by_percent = by_percent

    def __call__(self, sample):
        if sample[self.elem].ndim == 3:
            raise ValueError('GenerateTrimap not implemented for multiple object per image.')
        _target = sample[self.elem]
        if np.max(_target) == 0:
            sample['trimap'] = np.zeros(_target.shape, dtype=_target.dtype)
        else:
            if self.by_percent:
                # generate trimap by mask percent
                factor = np.sqrt(len(np.where(_target > 0)[0]))
                low = max(3, int(factor / 30.0))
                high = low + 3
                k_size_e = random.choice(range(low, high))
                k_size_d = random.choice(range(low, high))
                iterations_e = 5
                iterations_d = 5
            else:
                # generate trimap by fixed value
                k_size_e = random.choice(range(5,15))
                iterations_e = 2
                k_size_d = random.choice(range(5,15))
                iterations_d = 2            
            kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size_e, k_size_e))
            kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size_d, k_size_d))
            eroded = cv2.erode(_target, kernel_e, iterations=iterations_e)
            dilated = cv2.dilate(_target, kernel_d, iterations=iterations_d)
            trimap = np.zeros_like(_target).astype(np.uint8)
            trimap.fill(128)
            trimap[eroded >= 1] = 255
            trimap[dilated <= 0] = 0
            sample['trimap'] = trimap

            # trimap = helpers.generate_trimap_with_gaussian(_target)
            # trimap = helpers.produce_trimap(_target)            
            sample['trimap'] = trimap

        return sample

    def __str__(self):
        return 'GenerateTrimap:(kernel='+str(self.kernel)+', elem='+str(self.elem)+')'


class ConcatInputs(object):

    def __init__(self, elems=('image', 'point')):
        self.elems = elems

    def __call__(self, sample):

        res = sample[self.elems[0]]

        for elem in self.elems[1:]:
            assert(sample[self.elems[0]].shape[:2] == sample[elem].shape[:2])

            # Check if third dimension is missing
            tmp = sample[elem]
            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            res = np.concatenate((res, tmp), axis=2)

        sample['concat'] = res

        return sample

    def __str__(self):
        return 'ConcatInputs:'+str(self.elems)


class CropFromMask(object):
    """
    Returns image cropped in bounding box from a given mask
    """
    def __init__(self, crop_elems=('image', 'gt'),
                 relax=0,
                 zero_pad=False,
                 crop=True,
                 use_roimasking=False,
                 is_matting=False):

        self.crop_elems = crop_elems
        self.relax = relax
        self.zero_pad = zero_pad
        self.crop = crop
        self.use_roimasking = use_roimasking
        self.is_matting = is_matting

    def __call__(self, sample):
        if self.is_matting:
            _target = sample['alpha']
        else:
            _target = sample['gt']

        if _target.ndim == 2:
            _target = np.expand_dims(_target, axis=-1)
        for elem in self.crop_elems:
            if self.crop:# if cropped in bounding box
                _img = sample[elem]
                _crop = []
                if elem in ['gt', 'scribble', 'trimap', 'trimap_ori', 'alpha']:
                    if _img.ndim == 2:
                        _img = np.expand_dims(_img, axis=-1)
                    for k in range(0, _target.shape[-1]):
                        _tmp_img = _img[..., k]
                        _tmp_target = _target[..., k]
                        if self.relax > 0 and self.relax < 1:
                            target_ids = np.where(_tmp_target > 0)
                            if len(target_ids[0]) > 0 and len(target_ids[1]) > 0:
                                x_len = max(target_ids[1]) - min(target_ids[1]) + 1
                                y_len = max(target_ids[0]) - min(target_ids[0]) + 1
                                mask_shape_max = max(x_len, y_len)
                                mask_relax = max(20, int(self.relax * mask_shape_max))
                            else:
                                mask_relax = 0
                        else:
                            mask_relax = self.relax
                        if np.max(_target[..., k]) == 0:
                            _crop.append(np.zeros(_tmp_img.shape, dtype=_img.dtype))
                        else:
                            _crop.append(helpers.crop_from_mask(_tmp_img, _tmp_target, relax=mask_relax, 
                                            zero_pad=self.zero_pad, use_roimasking = self.use_roimasking))
                else:
                    for k in range(0, _target.shape[-1]):
                        if np.max(_target[..., k]) == 0:
                            _crop.append(np.zeros(_img.shape, dtype=_img.dtype))
                        else:
                            _tmp_target = _target[..., k]
                            if self.relax > 0 and self.relax < 1:
                                target_ids = np.where(_tmp_target > 0)
                                if len(target_ids) > 0:
                                    x_len = target_ids[1].max() - target_ids[1].min() + 1
                                    y_len = target_ids[0].max() - target_ids[0].min() + 1
                                    mask_shape_max = max(x_len, y_len)
                                    mask_relax = max(20, int(self.relax * mask_shape_max))
                                else:
                                    mask_relax = 0
                            else:
                                mask_relax = self.relax
                            _crop.append(helpers.crop_from_mask(_img, _tmp_target, relax=mask_relax, 
                                            zero_pad=self.zero_pad, use_roimasking = self.use_roimasking))
                if len(_crop) == 1:
                    if self.is_matting:
                        sample[elem] = _crop[0]
                    else:
                        sample['crop_' + elem] = _crop[0]
                else:
                    if self.is_matting:
                        sample[elem] = _crop
                    else:
                        sample['crop_' + elem] = _crop
            else:
                if not self.is_matting:
                    sample['crop_' + elem] = sample[elem]

            # adjust the alpha size field in test stage
            if self.is_matting and 'image' in self.crop_elems:
                sample['alpha_shape'] = sample['alpha'].shape
        # print('crop_scribble: %d' %len(np.where(sample['crop_scribble'] > 0)[0]))
        return sample

    def __str__(self):
        return 'CropFromMask:(crop=' + str(self.crop) + ', crop_elems=' +str(self.crop_elems)+\
               ', relax='+str(self.relax)+',zero_pad='+str(self.zero_pad)+')'


class ToImage(object):
    """
    Return the given elements between 0 and 255
    """
    def __init__(self, norm_elem=('image', ), custom_max=255.):
        self.norm_elem = norm_elem
        self.custom_max = custom_max

    def __call__(self, sample):
        if isinstance(self.norm_elem, tuple):
            for elem in self.norm_elem:
                tmp = sample[elem]
                sample[elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        else:
            tmp = sample[self.norm_elem]
            sample[self.norm_elem] = self.custom_max * (tmp - tmp.min()) / (tmp.max() - tmp.min() + 1e-10)
        return sample

    def __str__(self):
        return 'NormalizeImage:' + str(self.norm_elem)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            if 'meta' in elem:
                continue
            elif 'bbox' in elem:
                tmp = sample[elem]
                sample[elem] = torch.from_numpy(tmp)
                continue

            tmp = sample[elem]

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp)

        return sample

    def __str__(self):
        return 'ToTensor'


class MattingToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """

    def __init__(self, phase="test", trimap_type='standard', in_channels=6, trimap_one_hot=True):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.trimap_type=trimap_type
        self.phase = phase
        self.in_channels = in_channels
        self.trimap_one_hot = trimap_one_hot

    def __call__(self, sample):
        image, alpha, trimap = sample['image'], sample['alpha'], sample['trimap']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = sample['image']
        alpha = np.expand_dims(alpha.astype(np.float32), axis=0)
        trimap[trimap < 85] = 0
        trimap[trimap >= 170] = 2
        trimap[trimap >= 85] = 1

        if self.trimap_type == 'unified':
            trimap = helpers.unified_trimap_transform(trimap=trimap, sample_name=sample['meta']['image'], split_dir=sample['meta']['split_dir'])

        if self.phase == "train":
            # convert GBR images to RGB
            fg = sample['fg'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['fg'] = torch.from_numpy(fg).sub_(self.mean).div_(self.std)
            bg = sample['bg'][:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
            sample['bg'] = torch.from_numpy(bg).sub_(self.mean).div_(self.std)
            mask = sample['mask']
            mask = np.expand_dims(mask, axis=0)
            sample['mask'] = torch.from_numpy(mask)

            # del sample['image_name']
        else:
            h, w = image.shape[:2]
            if not (h % 32 == 0 and w % 32 == 0):
                target_h = 32 * ((h - 1) // 32 + 1)
                target_w = 32 * ((w - 1) // 32 + 1)
                pad_h = target_h - h
                pad_w = target_w - w
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                trimap = np.pad(trimap, ((0, pad_h), (0, pad_w)), mode="reflect")
                for elem in sample.keys():
                    if elem in ['extreme_points', 'in_points', 'out_points', 'scribble']:
                        sample[elem] = np.pad(sample[elem], ((0, pad_h), (0, pad_w)), mode="reflect")
        
        for elem in sample.keys():
            if elem in ['extreme_points', 'in_points', 'out_points', 'scribble']:
                tmp = sample[elem] / 255.
                if tmp.ndim == 2:
                    tmp = tmp[:, :, np.newaxis]
                tmp = tmp.transpose((2, 0, 1))
                sample[elem] = torch.from_numpy(tmp)            

        sample['alpha'], sample['trimap'] = torch.from_numpy(alpha), torch.from_numpy(trimap).to(torch.long)
        # normalize image
        image = image[:, :, ::-1].transpose((2, 0, 1)).astype(np.float32) / 255.
        sample['image'] = torch.from_numpy(image).sub_(self.mean).div_(self.std)

        # if self.in_channels == 4:
        #     sample['trimap'] = sample['trimap'][None, ...].float()
        # else:
        if self.trimap_one_hot:
            sample['trimap'] = F.one_hot(sample['trimap'], num_classes=3).permute(2, 0, 1).float()
        else:
            sample['trimap'] = sample['trimap'][None, ...].float()

        return sample


class AlignImg(object):
    """Align Image"""

    def __init__(self, elems=('crop_img', 'crop_gt'), align_stride=16.0, del_elem=True):
        self.align_elems = list(elems.keys())
        self.align_stride = align_stride
        self.del_elem = del_elem

    def __call__(self, sample):
        elems = list(sample.keys())
        for elem in elems:
            if 'meta' in elem or 'bbox' in elem or ('extreme_points_coord' in elem and elem not in self.resolutions):
                continue
            if elem in self.align_elems:
                w, h = sample[elem].shape[:2]
                new_w = int(np.ceil(w/float(self.align_stride))*self.align_stride)
                new_h = int(np.ceil(h/float(self.align_stride))*self.align_stride)
                if sample[elem].ndim == 3:
                    new_img = np.zeros((new_w, new_h, sample[elem].shape[-1]))
                else:
                    new_img = np.zeros((new_w, new_h))
                try:
                    new_img[:w, :h] = sample[elem]
                    sample[elem] = new_img.astype(np.float32)
                except Exception:
                    import IPython
                    IPython.embed()
            else:
                if self.del_elem:
                    del sample[elem]
        return sample
    
    def __str__(self):
        return 'AlignImages'+str(self.align_elems)


class interactiveMattingTransform(object):
    # modified from transform.py of dingrunyu
    def __init__(self, channel, no_crop = False, diff_width = False,\
                 relax_crop = 50, zero_pad_crop = True, use_iogpoints = False,\
                 use_roimasking = False, use_trimap = False, use_bbox = False,\
                 use_in_point = False, use_iogdextr = False, use_extreme_points = False,\
                 use_scribble = False, rotate_degree = 30, scale = [0.8, 1.25], shear = 10,\
                 flip = 0.5, crop_size = 512, trimap_type = 'standard', mask_type = 'alpha',\
                 bbox_type = 'area', trimap_one_hot=True):
        self.channel = channel
        self.no_crop = no_crop
        self.diff_width = diff_width
        self.relax_crop = relax_crop
        self.zero_pad_crop = zero_pad_crop
        self.rotate_degree = rotate_degree
        self.scale = scale
        self.shear = shear
        self.flip = flip
        self.crop_size = crop_size
        self.trimap_type = trimap_type
        self.mask_type = mask_type
        self.bbox_type = bbox_type
        self.use_roimasking = use_roimasking
        self.trimap_one_hot = trimap_one_hot

        self.use_trimap = use_trimap
        self.use_extreme_points = use_extreme_points
        self.use_bbox = use_bbox
        self.use_in_point = use_in_point
        self.use_iogpoints = use_iogpoints
        self.use_iogdextr = use_iogdextr
        self.use_scribble = use_scribble

    def getTrainTransform(self):
        transform_tr = [
            RandomAffine(degrees=self.rotate_degree, scale=self.scale, shear=self.shear, flip=self.flip),  
            GenTrimap(),
            RandomCrop(output_size=(self.crop_size, self.crop_size)),
            RandomJitter(),
            Composite(),
            GenMask(mask_type=self.mask_type),
            MattingToTensor(phase="train", trimap_type=self.trimap_type, in_channels=self.channel, trimap_one_hot=self.trimap_one_hot)]

        tr_ep = ExtremePoints(sigma=10, pert=5, elem='alpha')
        tr_in = IOGPoints(sigma=10, pert=5, elem='alpha', p_type='in')
        tr_out = IOGPoints(sigma=10, pert=5, elem='alpha', p_type='out')
        tr_bbox = OutPoints(sigma=10, pert=5, elem='alpha')
        tr_crop = CropFromMask(crop_elems=('alpha', 'fg'), \
                relax=self.relax_crop, zero_pad=self.zero_pad_crop, \
                    crop=False if self.no_crop else True, use_roimasking = self.use_roimasking,\
                    is_matting=True)
        tr_scribble = GenerateScribble(elem='alpha')

        if not self.no_crop:
            transform_tr.insert(0, tr_crop)

        if self.channel == 5 and self.use_iogpoints:
            print('Use foreground/background points')
            transform_tr.insert(-3, tr_in)
            transform_tr.insert(-3, tr_out)
            transform_tr.insert(-3, ToImage(norm_elem=('in_points', 'out_points')))

        elif self.channel == 6 and self.use_trimap:
            print('Use trimap (one-hot)')

        elif self.channel == 4 and self.use_trimap:
            print('Use trimap')

        elif self.channel == 4 and self.use_bbox:
            print('Use bounding box')
            if self.bbox_type == 'points':
                transform_tr.insert(-3, tr_out)
            elif self.bbox_type == 'area':
                transform_tr.insert(-3, tr_bbox)
            else:
                raise RuntimeError('Wrong bbox type.')
            transform_tr.insert(-3, ToImage(norm_elem=('out_points')))

        elif self.channel == 4 and self.use_in_point:
            print('Use inside point')
            transform_tr.insert(-3, tr_in)
            transform_tr.insert(-3, ToImage(norm_elem=('in_points')))

        elif self.channel == 4 and self.use_extreme_points:
            print('Use extreme points')
            transform_tr.insert(-3, tr_ep)
            transform_tr.insert(-3, ToImage(norm_elem='extreme_points'))

        elif self.channel == 4 and self.use_scribble:
            print('Use scribble')
            transform_tr.insert(-3, tr_scribble)
            transform_tr.insert(-3, ToImage(norm_elem='scribble'))

        elif self.channel == 3:
            print('Use no annotation')
        
        else:
            raise NotImplementedError('Wrong interactive method.')

        print([str(tran) for tran in transform_tr])
        return transforms.Compose(transform_tr)

    def getTestTransform(self, reserveGT = False):
        transform_ts = [
            MattingToTensor(phase="test", in_channels=self.channel, trimap_one_hot=self.trimap_one_hot)]

        tr_ep = ExtremePoints(sigma=10, pert=5, elem='alpha')
        tr_in = IOGPoints(sigma=10, pert=5, elem='alpha', p_type='in')
        tr_out = IOGPoints(sigma=10, pert=5, elem='alpha', p_type='out')
        tr_bbox = OutPoints(sigma=10, pert=5, elem='alpha')
        tr_crop = CropFromMask(crop_elems=('image', 'alpha', 'trimap'), \
                relax=self.relax_crop, zero_pad=self.zero_pad_crop, \
                    crop=False if self.no_crop else True, use_roimasking = self.use_roimasking,\
                    is_matting=True)
        tr_scribble = GenerateScribble(elem='alpha')

        if not self.no_crop:
            transform_ts.insert(0, tr_crop)

        if self.channel == 5 and self.use_iogpoints:
            print('Use foreground/background points')
            transform_ts.insert(-1, tr_in)
            transform_ts.insert(-1, tr_out)
            transform_ts.insert(-1, ToImage(norm_elem=('in_points', 'out_points')))

        elif self.channel == 6 and self.use_trimap:
            print('Use trimap (one-hot)')

        elif self.channel == 4 and self.use_trimap:
            print('Use trimap')

        elif self.channel == 4 and self.use_bbox:
            print('Use bounding box')
            if self.bbox_type == 'points':
                transform_ts.insert(-1, tr_out)
            elif self.bbox_type == 'area':
                transform_ts.insert(-1, tr_bbox)
            else:
                raise RuntimeError('Wrong bbox type.')
            transform_ts.insert(-1, ToImage(norm_elem=('out_points')))

        elif self.channel == 4 and self.use_in_point:
            print('Use inside point')
            transform_ts.insert(-1, tr_in)
            transform_ts.insert(-1, ToImage(norm_elem=('in_points')))

        elif self.channel == 4 and self.use_extreme_points:
            print('Use extreme points')
            transform_ts.insert(-1, tr_ep)
            transform_ts.insert(-1, ToImage(norm_elem='extreme_points'))

        elif self.channel == 4 and self.use_scribble:
            print('Use scribble')
            transform_ts.insert(-1, tr_scribble)
            transform_ts.insert(-1, ToImage(norm_elem='scribble'))
            
        elif self.channel == 3:
            print('Use no annotation')

        else:
            raise NotImplementedError('Wrong interactive method.')
   
        print([str(tran) for tran in transform_ts])
        return transforms.Compose(transform_ts)


if __name__ == "__main__":
    from logger import logger
    ScribbleTransform.getTrainTransform(logger)
    ScribbleTransform.getTestTransform(logger)
