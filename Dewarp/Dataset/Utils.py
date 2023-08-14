import random

import albumentations as A
import cv2
import numpy as np
import torch


IMG_SIZE = [488, 712]
GRID_SIZE = [45, 31]
class SafeHorizontalFlip(A.HorizontalFlip):
    """
    Horizontal Flip that changes the order of the keypoints so that the top left one remains in the top left position.
    """

    def __init__(self, gridsize=GRID_SIZE, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.gridsize = gridsize

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = super().apply_to_keypoints(keypoints, **params)

        keypoints = np.array(keypoints).reshape(*self.gridsize, -1)[:, ::-1, :]
        keypoints = keypoints.reshape(np.product(self.gridsize), -1)
        return keypoints

    def get_transform_init_args_names(self):
        return ("gridsize",)


class SafePerspective(A.Perspective):

    def __init__(
        self,
        scale=(0.05, 0.1),
        keep_size=True,
        pad_mode=cv2.BORDER_CONSTANT,
        pad_val=0,
        mask_pad_val=0,
        fit_output=False,
        interpolation=cv2.INTER_LINEAR,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(
            scale,
            keep_size,
            pad_mode,
            pad_val,
            mask_pad_val,
            fit_output,
            interpolation,
            always_apply,
            p,
        )

    @property
    def targets_as_params(self):
        return ["image", "keypoints"]

    def get_params_dependent_on_targets(self, params):
        h, w = params["image"].shape[:2]
        keypoints = np.array(params["keypoints"])[:, :2] / np.array([w, h])
        left = np.min(keypoints[:, 0])
        right = np.max(keypoints[:, 0])
        top = np.min(keypoints[:, 1])
        bottom = np.max(keypoints[:, 1])

        points = np.zeros([4, 2])
        # Top Left point
        points[0, 0] = A.random_utils.uniform(0, max(left - 0.01, left / 2))
        points[0, 1] = A.random_utils.uniform(0, max(top - 0.01, top / 2))
        # Top right point
        points[1, 0] = A.random_utils.uniform(min(right + 0.01, (right + 1) / 2), 1)
        points[1, 1] = A.random_utils.uniform(0, max(top - 0.01, top / 2))
        # Bottom Right point
        points[2, 0] = A.random_utils.uniform(min(right + 0.01, (right + 1) / 2), 1)
        points[2, 1] = A.random_utils.uniform(min(bottom + 0.01, (bottom + 1) / 2), 1)
        # Bottom Left point
        points[3, 0] = A.random_utils.uniform(0, max(left - 0.01, left / 2))
        points[3, 1] = A.random_utils.uniform(min(bottom + 0.01, (bottom + 1) / 2), 1)

        points[:, 0] *= w
        points[:, 1] *= h

        # Obtain a consistent order of the points and unpack them individually.
        # Warning: don't just do (tl, tr, br, bl) = _order_points(...)
        # here, because the reordered points is used further below.
        points = self._order_points(points)
        tl, tr, br, bl = points

        # compute the width of the new image, which will be the
        # maximum distance between bottom-right and bottom-left
        # x-coordiates or the top-right and top-left x-coordinates
        min_width = None
        max_width = None
        while min_width is None or min_width < 2:
            width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            max_width = int(max(width_top, width_bottom))
            min_width = int(min(width_top, width_bottom))
            if min_width < 2:
                step_size = (2 - min_width) / 2
                tl[0] -= step_size
                tr[0] += step_size
                bl[0] -= step_size
                br[0] += step_size

        # compute the height of the new image, which will be the maximum distance between the top-right
        # and bottom-right y-coordinates or the top-left and bottom-left y-coordinates
        min_height = None
        max_height = None
        while min_height is None or min_height < 2:
            height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            max_height = int(max(height_right, height_left))
            min_height = int(min(height_right, height_left))
            if min_height < 2:
                step_size = (2 - min_height) / 2
                tl[1] -= step_size
                tr[1] -= step_size
                bl[1] += step_size
                br[1] += step_size

        # now that we have the dimensions of the new image, construct
        # the set of destination points to obtain a "birds eye view",
        # (i.e. top-down view) of the image, again specifying points
        # in the top-left, top-right, bottom-right, and bottom-left order
        # do not use width-1 or height-1 here, as for e.g. width=3, height=2
        # the bottom right coordinate is at (3.0, 2.0) and not (2.0, 1.0)
        dst = np.array(
            [[0, 0], [max_width, 0], [max_width, max_height], [0, max_height]],
            dtype=np.float32,
        )

        # compute the perspective transform matrix and then apply it
        m = cv2.getPerspectiveTransform(points, dst)

        if self.fit_output:
            m, max_width, max_height = self._expand_transform(m, (h, w))

        return {
            "matrix": m,
            "max_height": max_height,
            "max_width": max_width,
            "interpolation": self.interpolation,
        }
def get_appearance_transform(transform_types):
    transforms = []
    if "shadow" in transform_types:
        transforms.append(A.RandomShadow(p=0.1))
    if "blur" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.Defocus(p=5),
                    A.Downscale(p=15, interpolation=cv2.INTER_LINEAR),
                    A.GaussianBlur(p=65),
                    A.MedianBlur(p=15),
                ],
                p=0.75,
            )
        )
    if "visual" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ToSepia(p=15),
                    A.ToGray(p=20),
                    A.Equalize(p=15),
                    A.Sharpen(p=20),
                ],
                p=0.5,
            )
        )
    if "noise" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.GaussNoise(var_limit=(10.0, 20.0), p=70),
                    A.ISONoise(intensity=(0.1, 0.25), p=30),
                ],
                p=0.6,
            )
        )
    if "color" in transform_types:
        transforms.append(
            A.OneOf(
                transforms=[
                    A.ColorJitter(p=5),
                    A.HueSaturationValue(p=10),
                    A.RandomBrightnessContrast(brightness_limit=[-0.05, 0.25], p=85),
                ],
                p=0.95,
            )
        )

    return A.Compose(transforms=transforms)


def get_geometric_transform(transform_types, gridsize):
    """
    Returns an albumentation compose augmentation.

    transform_type is a list containing types of geometric data augmentation to use.
    Possible augmentations are 'rotate', 'flip' and 'perspective'.
    """

    transforms = []
    if "rotate" in transform_types:
        transforms.append(
            A.SafeRotate(
                limit=[-30, 30],
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REPLICATE,
                p=0.5,
            )
        )
    if "flip" in transform_types:
        transforms.append(SafeHorizontalFlip(gridsize=gridsize, p=0.25))

    if "perspective" in transform_types:
        transforms.append(SafePerspective(p=0.5))

    return A.ReplayCompose(
        transforms=transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def crop_image_tight(img, grid2D):
    """
    Crops the image tightly around the keypoints in grid2D.
    This function creates a tight crop around the document in the image.
    """
    size = img.shape

    minx = np.floor(np.amin(grid2D[0, :, :])).astype(int)
    maxx = np.ceil(np.amax(grid2D[0, :, :])).astype(int)
    miny = np.floor(np.amin(grid2D[1, :, :])).astype(int)
    maxy = np.ceil(np.amax(grid2D[1, :, :])).astype(int)
    s = 20
    s = min(min(s, minx), miny)  # s shouldn't be smaller than actually available natural padding is
    s = min(min(s, size[1] - 1 - maxx), size[0] - 1 - maxy)

    # Crop the image slightly larger than necessary
    img = img[miny - s : maxy + s, minx - s : maxx + s, :]
    cx1 = random.randint(0, max(s - 5, 1))
    cx2 = random.randint(0, max(s - 5, 1)) + 1
    cy1 = random.randint(0, max(s - 5, 1))
    cy2 = random.randint(0, max(s - 5, 1)) + 1

    img = img[cy1:-cy2, cx1:-cx2, :]
    top = miny - s + cy1
    bot = size[0] - maxy - s + cy2
    left = minx - s + cx1
    right = size[1] - maxx - s + cx2
    return img, top, bot, left, right


class BaseDataset(torch.utils.data.Dataset):
    """
    Base torch dataset class for all unwarping dataset.
    """

    def __init__(
        self,
        data_path,
        appearance_augmentation=[],
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
    ) -> None:
        super().__init__()

        self.dataroot = data_path
        self.img_size = img_size
        self.grid_size = grid_size
        self.normalize_3Dgrid = True

        self.appearance_transform = get_appearance_transform(appearance_augmentation)

        self.all_samples = []

    def __len__(self):
        return len(self.all_samples)

    def crop_tight(self, img_RGB, grid2D):
        # The incoming grid2D array is expressed in pixel coordinates (resolution of img_RGB before crop/resize)
        size = img_RGB.shape
        img, top, bot, left, right = crop_image_tight(img_RGB, grid2D)
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        grid2D[0, :, :] = (grid2D[0, :, :] - left) / (size[1] - left - right)
        grid2D[1, :, :] = (grid2D[1, :, :] - top) / (size[0] - top - bot)
        grid2D = (grid2D * 2.0) - 1.0

        return img, grid2D