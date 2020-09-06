from PIL import Image
import random
import numpy as np
import cv2
import math

def rotate_image(img, angle):
    """
    angle: 旋转的角度
    crop: 是否需要进行裁剪，布尔向量
    """
    img = np.array(img)
    h,w = img.shape[:2]
    # 旋转角度的周期是360°
    angle %= 360
    # 计算仿射变换矩阵
    M_rotation = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotation, (w, h))
    # 如果需要去除黑边
    # 裁剪角度的等效周期是180°
    angle_crop = angle % 180
    if angle > 90:
        angle_crop = 180 - angle_crop
    # 转化角度为弧度
    theta = angle_crop * np.pi / 180
    # 计算高宽比
    hw_ratio = float(h) / float(w)
    # 计算裁剪边长系数的分子项
    tan_theta = np.tan(theta)
    numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)

    # 计算分母中和高宽比相关的项
    r = hw_ratio if h > w else 1 / hw_ratio
    # 计算分母项
    denominator = r * tan_theta + 1
    # 最终的边长系数
    crop_mult = numerator / denominator

    # 得到裁剪区域
    w_crop = int(crop_mult * w)
    h_crop = int(crop_mult * h)
    x0 = int((w - w_crop) / 2)
    y0 = int((h - h_crop) / 2)
    img_rotated = img_rotated[x0:x0 + w_crop, y0:y0 + h_crop, :]

    return img_rotated



class Resize(object):
    """Resize the input PIL Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)


class RandomCrop(object):
    """ random crop image """

    def __init__(self, size, scale=None, ratio=None):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

    def _get_params(self,img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        i, j, h, w = self._get_params(img, self.size)
        return img.crop((j, i, j + w, i + h))


class RandomHorizontalFlip(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img

class RandomRotate(object):
    def __init__(self,p=0.5):
        self.p=p
    def __call__(self,img):
        if random.random() < self.p:
            random_angle = np.random.randint(-30, 30)
            img_rotate = rotate_image(img,random_angle)
            return Image.fromarray(img_rotate)

        else:
            return img

class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.4, r1=0.3,
                 mean=[0., 0., 0.]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            img = np.array(img)
            area = img.shape[1] * img.shape[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                x1 = random.randint(0, img.shape[1] - h)
                y1 = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                return Image.fromarray(img)
        return Image.fromarray(img)

class Normalize(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw'):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        self.mean = np.array(mean).astype('float32')
        self.std = np.array(std).astype('float32')

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"
        return (img.astype('float32') * self.scale - self.mean) / self.std


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))
