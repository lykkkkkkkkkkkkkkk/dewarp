import json
import math
import os
import warnings
from os.path import join as pjoin
import torch.nn.functional as F
import cv2
import h5py as h5
import numpy as np
import torch

from Utils import BaseDataset, get_geometric_transform,GRID_SIZE,IMG_SIZE


class UVDocDataset(BaseDataset):
    def __init__(
        self,
        data_path="./data/UVdoc",
        appearance_augmentation=[],
        geometric_augmentations=[],
        grid_size=GRID_SIZE,
    ) -> None:
        super().__init__(
            data_path=data_path,
            appearance_augmentation=appearance_augmentation,
            img_size=IMG_SIZE,
            grid_size=grid_size,
        )
        self.original_grid_size = (89, 61)  # size of the captured data
        self.grid3d_normalization = (0.11433014, -0.12551452, 0.12401487, -0.12401487, 0.1952378, -0.1952378)
        self.geometric_transform = get_geometric_transform(geometric_augmentations, gridsize=self.original_grid_size)

        self.all_samples = [x[:-4] for x in os.listdir(pjoin(self.dataroot, "img")) if x.endswith(".png")]

    def __getitem__(self, index):
        # Get all paths
        sample_id = self.all_samples[index]
        with open(pjoin(self.dataroot, "metadata_sample", f"{sample_id}.json"), "r") as f:
            sample_name = json.load(f)["geom_name"]
        img_path = pjoin(self.dataroot, "img", f"{sample_id}.png")
        grid2D_path = pjoin(self.dataroot, "grid2d", f"{sample_name}.mat")
        grid3D_path = pjoin(self.dataroot, "grid3d", f"{sample_name}.mat")

        # Load 2D grid, 3D grid and image. Normalize 3D grid
        with h5.File(grid2D_path, "r") as file:
            grid2D_ = np.array(file["grid2d"][:].T.transpose(2, 0, 1))  # scale in range of img resolution

        with h5.File(grid3D_path, "r") as file:
            grid3D = np.array(file["grid3d"][:].T)

        if self.normalize_3Dgrid:  # scale grid3D to [0,1], based on stats computed over the entire dataset
            xmx, xmn, ymx, ymn, zmx, zmn = self.grid3d_normalization
            grid3D[:, :, 0] = (grid3D[:, :, 0] - xmn) / (xmx - xmn)
            grid3D[:, :, 1] = (grid3D[:, :, 1] - ymn) / (ymx - ymn)
            grid3D[:, :, 2] = (grid3D[:, :, 2] - zmn) / (zmx - zmn)
            grid3D = np.array(grid3D, dtype=np.float32)
        grid3D = torch.from_numpy(grid3D.transpose(2, 0, 1))

        img_RGB_ = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # Pixel-wise augmentation
        img_RGB_ = self.appearance_transform(image=img_RGB_)["image"]

        # Geometric Augmentations
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            transformed = self.geometric_transform(
                image=img_RGB_,
                keypoints=grid2D_.transpose(1, 2, 0).reshape(-1, 2),
            )
            img_RGB_ = transformed["image"]

            grid2D_ = np.array(transformed["keypoints"]).reshape(*self.original_grid_size, 2).transpose(2, 0, 1)

            flipped = False
            for x in transformed["replay"]["transforms"]:
                if "SafeHorizontalFlip" in x["__class_fullname__"]:
                    flipped = x["applied"]
            if flipped:
                grid3D[1] = 1 - grid3D[1]
                grid3D = torch.flip(grid3D, dims=(2,))

        # Tight crop
        grid2Dtmp = grid2D_
        img_RGB, grid2D = self.crop_tight(img_RGB_, grid2Dtmp)

        # Subsample grids to desired resolution
        row_sampling_factor = math.ceil(self.original_grid_size[0] / self.grid_size[0])
        col_sampling_factor = math.ceil(self.original_grid_size[1] / self.grid_size[1])
        grid3D = grid3D[:, ::row_sampling_factor, ::col_sampling_factor]
        grid2D = grid2D[:, ::row_sampling_factor, ::col_sampling_factor]
        grid2D = torch.from_numpy(grid2D).float()

        # Unwarp the image according to grid
        img_RGB_unwarped = bilinear_unwarping(img_RGB.unsqueeze(0), grid2D.unsqueeze(0), self.img_size).squeeze()

        return (
            img_RGB.float() / 255.0,
            img_RGB_unwarped.float() / 255.0,
            grid2D,
            grid3D,
        )
class doc3DDataset(BaseDataset):

    def __init__(
        self,
        data_path="./data/doc3D",
        split="train",
        appearance_augmentation=[],
        grid_size=GRID_SIZE,
    ):
        super().__init__(
            data_path=data_path,
            appearance_augmentation=appearance_augmentation,
            img_size=IMG_SIZE,
            grid_size=grid_size,
        )
        self.grid3d_normalization = (1.2539363, -1.2442188, 1.2396319, -1.2289206, 0.6436657, -0.67492497)

        if split == "train":
            path = pjoin(self.dataroot, "traindoc.txt")
        elif split == "val":
            path = pjoin(self.dataroot, "valdoc3D.txt")

        with open(path, "r") as files:
            file_list = tuple(files)
        self.all_samples = np.array([id_.rstrip() for id_ in file_list], dtype=np.string_)

    def __getitem__(self, index):
        # Get all paths
        im_name = self.all_samples[index].decode("UTF-8")
        img_path = pjoin(self.dataroot, "img", im_name + ".png")
        grid2D_path = pjoin(self.dataroot, "grid2D", im_name + ".mat")
        grid3D_path = pjoin(self.dataroot, "grid3D", im_name + ".mat")
        bm_path = pjoin(self.dataroot, "bm", im_name + ".mat")

        # Load 2D grid, 3D grid and image. Normalize 3D grid
        with h5.File(grid2D_path, "r") as file:
            grid2D_ = np.array(file["grid2D"][:].T.transpose(2, 0, 1))  # scale in range of img resolution

        with h5.File(grid3D_path, "r") as file:
            grid3D = np.array(file["grid3D"][:].T)

        if self.normalize_3Dgrid:  # scale grid3D to [0,1], based on stats computed over the entire dataset
            xmx, xmn, ymx, ymn, zmx, zmn = self.grid3d_normalization
            grid3D[:, :, 0] = (grid3D[:, :, 0] - zmn) / (zmx - zmn)
            grid3D[:, :, 1] = (grid3D[:, :, 1] - ymn) / (ymx - ymn)
            grid3D[:, :, 2] = (grid3D[:, :, 2] - xmn) / (xmx - xmn)
            grid3D = np.array(grid3D, dtype=np.float32)
        grid3D[:, :, 1] = grid3D[:, :, 1][:, ::-1]
        grid3D[:, :, 1] = 1 - grid3D[:, :, 1]
        grid3D = torch.from_numpy(grid3D.transpose(2, 0, 1))

        img_RGB_ = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        # Pixel-wise augmentation
        img_RGB_ = self.appearance_transform(image=img_RGB_)["image"]

        # Create unwarped image according to the backward mapping (first load the backward mapping)
        with h5.File(bm_path, "r") as file:
            bm = np.array(file["bm"][:].T.transpose(2, 0, 1))
        bm = ((bm / 448) - 0.5) * 2.0
        bm = torch.from_numpy(bm).float()

        img_RGB_unwarped = bilinear_unwarping(
            torch.from_numpy(img_RGB_.transpose(2, 0, 1)).float().unsqueeze(0),
            bm.unsqueeze(0),
            self.img_size,
        ).squeeze()

        # Tight crop
        grid2Dtmp = grid2D_
        img_RGB, grid2D = self.crop_tight(img_RGB_, grid2Dtmp)

        # Convert 2D grid to torch tensor
        grid2D = torch.from_numpy(grid2D).float()

        return (
            img_RGB.float() / 255.0,
            img_RGB_unwarped.float() / 255.0,
            grid2D,
            grid3D,
        )
class mixDataset(torch.utils.data.Dataset):
    """
    Class to use both UVDoc and Doc3D datasets at the same time.
    """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, ii):
        if len(self.datasets[0]) < len(self.datasets[1]):
            len_shortest = len(self.datasets[0])
            i_shortest = ii % len_shortest
            return self.datasets[0][i_shortest], self.datasets[1][ii]
        else:
            len_shortest = len(self.datasets[1])
            jj = ii % len_shortest
            return self.datasets[0][ii], self.datasets[1][jj]

    def __len__(self):
        return max(len(d) for d in self.datasets)
def bilinear_unwarping(warped_img, point_positions, img_size):
    upsampled_grid = F.interpolate(
        point_positions, size=(img_size[1], img_size[0]), mode="bilinear", align_corners=True
    )
    unwarped_img = F.grid_sample(warped_img, upsampled_grid.transpose(1, 2).transpose(2, 3), align_corners=True)

    return unwarped_img


def bilinear_unwarping_from_numpy(warped_img, point_positions, img_size):
    warped_img = torch.unsqueeze(torch.from_numpy(warped_img.transpose(2, 0, 1)).float(), dim=0)
    point_positions = torch.unsqueeze(torch.from_numpy(point_positions.transpose(2, 0, 1)).float(), dim=0)

    unwarped_img = bilinear_unwarping(warped_img, point_positions, img_size)

    unwarped_img = unwarped_img[0].numpy().transpose(1, 2, 0)
    return unwarped_img