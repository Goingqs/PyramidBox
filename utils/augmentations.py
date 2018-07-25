import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random
import math
cv2.setNumThreads(0)

def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image /= 256
        image -= self.mean
        image /= self.std
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=640):
        self.size = size
        
    def __call__(self, image, boxes=None, labels=None):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, (self.size,self.size),interpolation=interp_method)
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 1] *= random.uniform(self.lower, self.upper)
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class RandomBaiduCrop(object):
    """Crop
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        
        self.mean = np.array([104, 117, 123],dtype=np.float32)
        self.maxSize = 12000    #max size
        self.infDistance = 9999999

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape

        random_counter = 0

        boxArea = (boxes[:,2] - boxes[:,0] + 1) * (boxes[:,3] - boxes[:,1] + 1)
        #argsort = np.argsort(boxArea)
        #rand_idx = random.randint(min(len(argsort),6))
        #print('rand idx',rand_idx)
        rand_idx = random.randint(len(boxArea))
        rand_Side = boxArea[rand_idx] ** 0.5
        #rand_Side = min(boxes[rand_idx,2] - boxes[rand_idx,0] + 1, boxes[rand_idx,3] - boxes[rand_idx,1] + 1)
        
        anchors = [16,32,64,128,256,512]
        distance = self.infDistance
        anchor_idx = 5
        for i,anchor in enumerate(anchors):
            if abs(anchor-rand_Side) < distance:
                distance = abs(anchor-rand_Side)
                anchor_idx = i

        target_anchor = random.choice(anchors[0:min(anchor_idx+1,5)+1])
        ratio = float(target_anchor) / rand_Side
        ratio = ratio * (2**random.uniform(-1,1))
        
        if int(height * ratio * width * ratio) > self.maxSize*self.maxSize:
            ratio = (self.maxSize*self.maxSize/(height*width))**0.5


        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)

        boxes[:,0] *= ratio
        boxes[:,1] *= ratio
        boxes[:,2] *= ratio
        boxes[:,3] *= ratio

        height, width, _ = image.shape

        sample_boxes = []
        
        xmin = boxes[rand_idx,0]
        ymin = boxes[rand_idx,1]
        bw = (boxes[rand_idx,2] - boxes[rand_idx,0] + 1)
        bh = (boxes[rand_idx,3] - boxes[rand_idx,1] + 1)

        w = h = 640

        for _ in range(50):
            if w < max(height,width):
                if bw <= w:
                    w_off = random.uniform(xmin + bw - w, xmin)
                else:
                    w_off = random.uniform(xmin, xmin + bw - w)

                if bh <= h:
                    h_off = random.uniform(ymin + bh - h, ymin)
                else:
                    h_off = random.uniform(ymin, ymin + bh -h)
            else:
                w_off = random.uniform(width - w, 0)
                h_off = random.uniform(height - h, 0)

            w_off = math.floor(w_off)
            h_off = math.floor(h_off)

            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(w_off), int(h_off), int(w_off+w), int(h_off+h)])


            # keep overlap with gt box IF center in sampled patch
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            # mask in all gt boxes that above and to the left of centers
            m1 = (rect[0] <= boxes[:, 0]) * (rect[1] <= boxes[:, 1])
            # mask in all gt boxes that under and to the right of centers
            m2 = (rect[2] >= boxes[:, 2]) * (rect[3] >= boxes[:, 3])
            # mask in that both m1 and m2 are true
            mask = m1 * m2

            overlap = jaccard_numpy(boxes,rect)
            # have any valid boxes? try again if not
            if not mask.any() and not overlap.max() > 0.7:
                continue
            else:
                sample_boxes.append(rect)
                

        if len(sample_boxes) > 0:
            choice_idx = random.randint(len(sample_boxes))
            choice_box = sample_boxes[choice_idx]
            #print('crop the box :',choice_box)
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            m1 = (choice_box[0] < centers[:, 0]) * (choice_box[1] < centers[:, 1])
            m2 = (choice_box[2] > centers[:, 0]) * (choice_box[3] > centers[:, 1])
            mask = m1 * m2
            current_boxes = boxes[mask, :].copy()
            current_labels = labels[mask]
            current_boxes[:, :2] -= choice_box[:2]
            current_boxes[:, 2:] -= choice_box[:2]
            if choice_box[0] < 0 or choice_box[1] < 0:
                new_img_width = width if choice_box[0] >=0 else width-choice_box[0]
                new_img_height = height if choice_box[1] >=0 else height-choice_box[1]
                image_pad = np.zeros((new_img_height,new_img_width,3),dtype=float)
                image_pad[:, :, :] = self.mean
                start_left = 0 if choice_box[0] >=0 else -choice_box[0]
                start_top = 0 if choice_box[1] >=0 else -choice_box[1]
                image_pad[start_top:,start_left:,:] = image

                choice_box_w = choice_box[2] - choice_box[0]
                choice_box_h = choice_box[3] - choice_box[1]

                start_left = choice_box[0] if choice_box[0] >=0 else 0
                start_top = choice_box[1] if choice_box[1] >=0 else 0
                end_right = start_left + choice_box_w
                end_bottom = start_top + choice_box_h
                current_image = image_pad[start_top:end_bottom,start_left:end_right,:].copy()
                return current_image, current_boxes, current_labels

            current_image = image[choice_box[1]:choice_box[3],choice_box[0]:choice_box[2],:].copy()
            return current_image, current_boxes, current_labels
        else:
            return image, boxes, labels

#--------------------------------



class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]

        self.rand_contrast = RandomContrast()
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        self.rand_hue = RandomHue()
        self.rand_saturation = RandomSaturation()

    def __call__(self, image, boxes, labels):

        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        im = image.copy()
        _convert(im, beta=random.uniform(-32, 32))
        if random.randint(2):
            _convert(im, alpha=random.uniform(0.5, 1.5))

            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            _convert(im[:, :, 1], alpha=random.uniform(0.5, 1.5))
            im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            tmp = im[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            im[:, :, 0] = tmp.astype(float)
            im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
            im = im.astype(float)

        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            _convert(im[:, :, 1], alpha=random.uniform(0.5, 1.5))
            im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

            im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            tmp = im[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            im[:, :, 0] = tmp.astype(float)
            im = cv2.cvtColor(im, cv2.COLOR_HSV2BGR)

            _convert(im, alpha=random.uniform(0.5, 1.5))

        return im,boxes,labels


class PyramidAugmentation(object):
    def __init__(self, size=640, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomBaiduCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            SubtractMeans(self.mean)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class PyramidNormalAugmentation(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485),std=(0.225, 0.224, 0.229)):
        self.mean = mean
        self.std = std
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            RandomBaiduCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.size),
            Normalize(self.mean,self.std)
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)
