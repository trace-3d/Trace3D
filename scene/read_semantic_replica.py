# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import math
import os

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import cv2
import numpy as np
from pathlib import Path

from utils.sh_utils import SH2RGB

from PIL import Image
from scene.gaussian_model import BasicPointCloud
import torch
import torchvision.transforms as transforms
from scene.config import DEFAULT_SAM_FOLDER
from scene.dataset_readers import SceneInfo, getNerfppNorm, storePly, fetchPly, glob_data,CameraInfo,read_points3D_binary

import time
#from egolifter/scene/dataset_readers/replica_semantic.py
def get_replica_semantic_intrisic(img_h:int = 480, img_w:int = 640):
    # replica dataset from semantic nerf used a fixed fov
    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    fx = img_w / 2.0 / math.tan(math.radians(hfov / 2.0))
    fy = fx
    cx = (img_w - 1.0) / 2.0
    cy = (img_h - 1.0) / 2.0
    return fx, fy, cx, cy


def read_semantic_ReplicaInfo(input_folder: str, image_stride:int = 1, sam_folder='origin'):
    input_folder = Path(input_folder)
    scene_name=os.path.basename(input_folder)
    traj_path = input_folder / "traj_w_c.txt"

    rgb_paths = glob.glob(str(input_folder / "rgb" / "rgb*.png"))
    depth_paths = glob_data(os.path.join(input_folder , "depth" , "depth*.png"))
    sam_paths = glob_data(os.path.join(input_folder , DEFAULT_SAM_FOLDER  ,sam_folder ,"*.npy"))

    assert len(rgb_paths) > 0, "No RGB images found at {}".format(str(input_folder / "results" / "frame*.jpg"))
    assert len(rgb_paths) == len(depth_paths), "Number of RGB and depth images must match"
    assert os.path.exists(traj_path), "Could not find camera trajectory at {}".format(traj_path)

    # Load the poses
    poses = np.loadtxt(traj_path, delimiter=" ").reshape(-1, 4, 4)
    assert len(poses) == len(rgb_paths), f"Number of poses ({len(poses)}) and number of images ({len(rgb_paths)}) must match"

    # Load the intrinsics
    img_h, img_w = 480, 640
    fx, fy, cx, cy = get_replica_semantic_intrisic(img_h, img_w)
    fovx = focal2fov(fx, img_w)
    fovy = focal2fov(fy, img_h)
    transf = transforms.ToTensor()

    train_camera_infos = []
    test_camera_infos = []

    # Split the train/valid sets according to Egolifter
    all_idx = np.arange(0, 900,image_stride)
    test_idx = np.arange(0, len(all_idx), 4) # 25% of the images are used for validation
    train_idx = np.setdiff1d(all_idx, all_idx[test_idx])
    reject_view = []  
    # from omniseg3D  
    if scene_name == 'office_1':
        reject_view = np.arange(474, 504)
        print(f"Rejecting views for office_1: {reject_view}")
    elif scene_name == 'office_4':
        reject_view = np.arange(618, 734)
        print(f"Rejecting views for office_4: {reject_view}")

    for idx in np.arange(0,900, image_stride):
        if idx in reject_view:
            continue
        rgb_path = os.path.join(input_folder , "rgb" , f"rgb_{idx}.png")
        gt_seg_path= os.path.join(input_folder , "semantic_instance",f"semantic_instance_{idx}.png")
        image = Image.open(rgb_path)
        image_name=rgb_path.split("/")[-1].split(".")[0]
        depth_path =os.path.join(input_folder , "depth" , f"depth_{idx}.png")# depth_paths[idx],gt depth
        depth = Image.open(depth_path)
        depth = (transf(depth)/1000)
        pose = poses[idx]
        
        id_masks = None
        sam_features = None
        if len(sam_paths) > 0 and (idx in train_idx):
            sam_mask = np.load(os.path.join(input_folder, DEFAULT_SAM_FOLDER, sam_folder,"rgb_{}.npy".format(idx)))
            if len(sam_mask.shape) == 3:
                N, H, W = sam_mask.shape
                sam_mask = torch.from_numpy(sam_mask).cuda()
                flat_mask = sam_mask.permute(1, 2, 0).reshape(-1, N)
                _, inverse_indices = torch.unique(flat_mask, return_inverse=True, dim = 0)
                id_masks = inverse_indices.view(H, W).cpu().numpy()
            else:
                id_masks = sam_mask
        try:
            sam_features = torch.load(os.path.join(input_folder, "sam_features", f"rgb_{idx}.pt"))
        except:
            pass

        gt_seg = None
        if os.path.exists(gt_seg_path):
            gt_seg=cv2.imread(gt_seg_path,cv2.IMREAD_UNCHANGED)

        pose = np.linalg.inv(pose)
        R = pose[:3,:3]
        R = R.T
        t = pose[:3,3]

        cam = CameraInfo(
            uid=idx,
            R=R,
            T=t,
            FovX=fovx,
            FovY=fovy,
            image=image,
            image_name=image_name,
            width= img_w, height=img_h,
            sam_mask=id_masks,
            instance_image=gt_seg,
            image_path=rgb_path,
            depth=depth,
            features=sam_features
        )
        if idx in train_idx:
            train_camera_infos.append(cam)
        elif idx in test_idx:
            test_camera_infos.append(cam)
        else:
            raise

    nerf_normalization = getNerfppNorm(train_camera_infos)

    ply_path = os.path.join(input_folder, "sparse/0/points3D.ply")
    bin_path = os.path.join(input_folder, "sparse/0/points3D.bin")
       
    if not os.path.exists(ply_path):
        if os.path.exists(bin_path):
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        else:
        # Since this data set has no colmap data, we start with random points
            num_pts = 10_000
            print(f"Generating random point cloud ({num_pts})...")
            
            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
            storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=str(ply_path)
    )

    return scene_info