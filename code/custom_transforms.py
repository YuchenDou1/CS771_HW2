import math
import random
import numpy as np
import torch

import cv2
import numbers
import collections
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from utils import resize_image

# default list of interpolations
_DEFAULT_INTERPOLATIONS = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC]

#################################################################################
# Solution for HW 1
#################################################################################


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> Compose([
        >>>     Scale(320),
        >>>     RandomSizedCrop(224),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        repr_str = ""
        for t in self.transforms:
            repr_str += t.__repr__() + "\n"
        return repr_str


class RandomHorizontalFlip(object):
    """Horizontally flip the given numpy array randomly
    (with a probability of 0.5).
    """

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be flipped.

        Returns:
            numpy array: Randomly flipped image
        """
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        return img

    def __repr__(self):
        return "Random Horizontal Flip"


#################################################################################
# You will need to fill in the missing code in these classes
#################################################################################
class Scale(object):
    """Rescale the input numpy array to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size, size * height / width)

        interpolations (list of int, optional): Desired interpolation.
        Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        Pass None during testing: always use CV2.INTER_LINEAR
    """

    def __init__(self, size, interpolations=_DEFAULT_INTERPOLATIONS):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2
        )
        self.size = size
        # use bilinear if interpolation is not specified
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        """
        Args:
            img (numpy array): Image to be scaled.

        Returns:
            numpy array: Rescaled image
        """
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        # scale the image
        if isinstance(self.size, int):
            h, w = img.shape[0], img.shape[1]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = int(self.size)
                oh = int(round(self.size * h / w))
                img = resize_image(img, (ow, oh), interpolation=interpolation)
            else:
                oh = int(self.size)
                ow = int(round(self.size * w / h))
                img = resize_image(img, (ow, oh), interpolation=interpolation)
            return img
        else:
            #################################################################################
            # Solution
            #################################################################################
            img = resize_image(img, self.size, interpolation=interpolation)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            return "Scale [Shortest side {:d}]".format(self.size)
        else:
            target_size = self.size
            return "Scale [Exact Size ({:d}, {:d})]".format(
                target_size[0], target_size[1]
            )


class CenterCrop(object):
    """
    Crop the center patch of a given image (numpy array)

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
    """

    def __init__(
        self,
        size,
    ):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]
        if isinstance(self.size, int):
            th, tw = self.size, self.size
        else:
            th, tw = self.size
        x1 = int(round((w - tw) / 2.0))
        y1 = int(round((h - th) / 2.0))
        img = img[y1 : y1 + th, x1 : x1 + tw]
        return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return "Random Crop" + "[Size ({:d}, {:d})]".format(
            target_size[0],
            target_size[1],
        )


class RandomSizedCrop(object):
    """Crop the given numpy array to random area and aspect ratio.

    A crop of random area of the original size and a random aspect ratio
    of the original aspect ratio is made. This crop is finally resized to given size.
    This is widely used as data augmentation for training image classification models

    Args:
        size (sequence or int): size of target image. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            output size will be (size, size).
        interpolations (list of int, optional): Desired interpolation.
        Default is ``CV2.INTER_NEAREST|CV2.INTER_LINEAR|CV2.INTER_CUBIC``
        area_range (list of int): range of the areas to sample from
        ratio_range (list of int): range of aspect ratio to sample from
        num_trials (int): number of sampling trials
    """

    def __init__(
        self,
        size,
        interpolations=_DEFAULT_INTERPOLATIONS,
        area_range=(0.25, 1.0),
        ratio_range=(0.8, 1.2),
        num_trials=10,
    ):
        self.size = size
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations
        self.num_trials = int(num_trials)
        self.area_range = area_range
        self.ratio_range = ratio_range

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]

        for attempt in range(self.num_trials):

            # sample target area / aspect ratio from area range and ratio range
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(self.area_range[0], self.area_range[1]) * area
            aspect_ratio = random.uniform(self.ratio_range[0], self.ratio_range[1])

            #################################################################################
            # Solution
            #################################################################################
            # compute the width and height
            # note that there are two possibilities
            # crop the image and resize to output size
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                w, h = h, w

            # crop the image
            if w <= img.shape[1] and h <= img.shape[0]:
                x1 = random.randint(0, img.shape[1] - w)
                y1 = random.randint(0, img.shape[0] - h)

                img = img[y1 : y1 + h, x1 : x1 + w]
                if isinstance(self.size, int):
                    img = resize_image(
                        img, (self.size, self.size), interpolation=interpolation
                    )
                else:
                    img = resize_image(img, self.size, interpolation=interpolation)
                return img

        # Fall back
        if isinstance(self.size, int):
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            #################################################################################
            # Solution
            #################################################################################
            # with a square sized output, the default is to crop the patch in the center
            # (after all trials fail)
            h, w = img.shape[0], img.shape[1]
            th, tw = self.size, self.size
            x1 = int(round((w - tw) / 2.0))
            y1 = int(round((h - th) / 2.0))
            img = img[y1 : y1 + th, x1 : x1 + tw]
            return img
        else:
            # with a pre-specified output size, the default crop is the image itself
            im_scale = Scale(self.size, interpolations=self.interpolations)
            img = im_scale(img)
            return img

    def __repr__(self):
        if isinstance(self.size, int):
            target_size = (self.size, self.size)
        else:
            target_size = self.size
        return (
            "Random Crop"
            + "[Size ({:d}, {:d}); Area {:.2f} - {:.2f}; Ratio {:.2f} - {:.2f}]".format(
                target_size[0],
                target_size[1],
                self.area_range[0],
                self.area_range[1],
                self.ratio_range[0],
                self.ratio_range[1],
            )
        )


class RandomColor(object):
    """Perturb color channels of a given image
    Sample alpha in the range of (-r, r) and multiply 1 + alpha to a color channel.
    The sampling is done independently for each channel.

    Args:
        color_range (float): range of color jitter ratio (-r ~ +r) max r = 1.0
    """

    def __init__(self, color_range):
        self.color_range = color_range
        self.color_lut = np.arange(256, dtype=np.float32)

    def __call__(self, img):
        #################################################################################
        # Solution
        #################################################################################
        # more efficient version using LUT
        color_lut = np.zeros([1, 256, 3], dtype=np.float32)
        for c in range(3):
            target_ratio = random.uniform(-self.color_range, self.color_range) + 1.0
            color_lut[0, :, c] = self.color_lut * target_ratio
        color_lut = np.clip(color_lut, 0, 255).astype(np.uint8)
        img = cv2.LUT(img, color_lut)
        return img

    def __repr__(self):
        return "Random Color [Range {:.2f} - {:.2f}]".format(
            1 - self.color_range, 1 + self.color_range
        )


class RandomRotate(object):
    """Rotate the given numpy array (around the image center) by a random degree.

    Args:
        degree_range (float): range of degree (-d ~ +d)
    """

    def __init__(self, degree_range, interpolations=_DEFAULT_INTERPOLATIONS):
        self.degree_range = degree_range
        if interpolations is None:
            interpolations = [cv2.INTER_LINEAR]
        assert isinstance(interpolations, Iterable)
        self.interpolations = interpolations

    def __call__(self, img):
        # sample interpolation method
        interpolation = random.sample(self.interpolations, 1)[0]
        # sample rotation
        degree = random.uniform(-self.degree_range, self.degree_range)
        # ignore small rotations
        if np.abs(degree) <= 1.0:
            return img

        #################################################################################
        # Solution
        #################################################################################
        # get the max area rectangular within the rotated image
        # ref: stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        h, w = img.shape[0], img.shape[1]
        side_long = float(max([h, w]))
        side_short = float(min([h, w]))

        # since the solutions for angle, -angle and pi-angle are all the same,
        # it suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a = np.abs(np.sin(np.pi * degree / 180))
        cos_a = np.abs(np.cos(np.pi * degree / 180))

        if side_short <= 2.0 * sin_a * cos_a * side_long:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            if w >= h:
                wr, hr = x / sin_a, x / cos_a
            else:
                wr, hr = x / cos_a, x / sin_a
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr = (w * cos_a - h * sin_a) / cos_2a
            hr = (h * cos_a - w * sin_a) / cos_2a

        rot_mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1.0)
        rot_mat[0, 2] += (wr - w) / 2.0
        rot_mat[1, 2] += (hr - h) / 2.0

        img = cv2.warpAffine(
            img, rot_mat, (int(round(wr)), int(round(hr))), flags=interpolation
        )

        return img

    def __repr__(self):
        return "Random Rotation [Range {:.2f} - {:.2f} Degree]".format(
            -self.degree_range, self.degree_range
        )


#################################################################################
# Additional helper functions
#################################################################################
class ToTensor(object):
    """Convert a ``numpy.ndarray`` image pair to tensor.

    Converts a numpy.ndarray (H x W x C) image in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img):
        assert isinstance(img, np.ndarray)
        # make ndarray normal
        img = img.copy()
        # convert image to tensor
        if img.ndim == 2:
            img = img[:, :, None]

        tensor_img = torch.from_numpy(img.transpose((2, 0, 1)))
        # backward compatibility
        if isinstance(tensor_img, torch.ByteTensor):
            return tensor_img.float().div(255)
        else:
            return tensor_img

    def __repr__(self):
        return "To Tensor()"


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
          respecitvely.
    """

    def __init__(self, mean, std, scale=1.0):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, tensor_img):
        # multiply scale -> subtract mean (per channel) -> divide by std (per channel)
        tensor_img.mul_(self.scale)
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor_img.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor_img.device)
        tensor_img.sub_(mean[:, None, None]).div_(std[:, None, None])
        return tensor_img

    def __repr__(self):
        return "Normalize" + "(mean={0}, std={1})".format(self.mean, self.std)


class Denormalize(object):
    """De-normalize an tensor image with mean and standard deviation.

    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel * std + mean

    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
          respecitvely.
    """

    def __init__(self, mean, std, scale=1.0):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, tensor_img):
        # multiply by std (per channel) -> add mean (per channel) -> divide by scale
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor_img.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor_img.device)
        tensor_img.mul_(std[:, None, None]).add_(mean[:, None, None])
        tensor_img.div_(self.scale)
        return tensor_img

    def __repr__(self):
        return "De-normalize" + "(mean={0}, std={1})".format(self.mean, self.std)
