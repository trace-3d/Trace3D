# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import math
import os

import json
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import cv2
import numpy as np
from pathlib import Path
from scene.cameras import Camera
# from natsort import natsorted
from utils.sh_utils import SH2RGB
from utils.general_utils import PILtoTorch
from PIL import Image
from scene.gaussian_model import BasicPointCloud
import torch
import torchvision.transforms as transforms

from scene.dataset_readers import SceneInfo, getNerfppNorm, storePly, fetchPly, glob_data,CameraInfo,read_points3D_binary

def get_replica_semantic_intrisic(img_h:int = 480, img_w:int = 640):
    # replica dataset from semantic nerf used a fixed fov
    hfov = 90
    # the pin-hole camera has the same value for fx and fy
    fx = img_w / 2.0 / math.tan(math.radians(hfov / 2.0))
    fy = fx
    cx = (img_w - 1.0) / 2.0
    cy = (img_h - 1.0) / 2.0
    return fx, fy, cx, cy

def read_semantic_ReplicaInfo(input_folder: str, image_stride:int = 1, eval=False,sam_folder='origin'):
    input_folder = Path(input_folder)

    traj_path = input_folder / "traj_w_c.txt"
    # points_path = input_folder / "rgb_cloud" / "pointcloud.ply"
    # rgb_paths = 
    # depth_paths = glob.glob(str(input_folder / "results" / "depth*.png"))
    # sam_paths = glob.glob(str(input_folder / "sam" / "*.npy"))

    rgb_paths = glob.glob(str(input_folder / "rgb" / "rgb*.png"))
    depth_paths = glob_data(os.path.join(input_folder , "depth" , "depth*.png"))
    sam_paths = glob_data(os.path.join(input_folder , "sam" ,sam_folder ,"*.npy"))
    #gt_seg_path=glob_data(os.path.join(input_folder , "semantic_instance","semantic_instance*.png"))
    sam_split_path=os.path.join(input_folder , "sam" , "split")

    assert len(rgb_paths) > 0, "No RGB images found at {}".format(str(input_folder / "results" / "frame*.jpg"))
    assert len(rgb_paths) == len(depth_paths), "Number of RGB and depth images must match"
    assert os.path.exists(traj_path), "Could not find camera trajectory at {}".format(traj_path)

    # Read point cloud
    # pointcloud = fetchPly(points_path, stride=pcd_stride)
    
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

    # Load the poses
    poses = np.loadtxt(traj_path, delimiter=" ").reshape(-1, 4, 4)
    assert len(poses) == len(rgb_paths), f"Number of poses ({len(poses)}) and number of images ({len(rgb_paths)}) must match"

    # Load the intrinsics
    img_h, img_w = 480, 640
    fx, fy, cx, cy = get_replica_semantic_intrisic(img_h, img_w)
    fovx = focal2fov(fx, img_w)
    fovy = focal2fov(fy, img_h)
    transf = transforms.ToTensor()
    # Load the images
    cam_list = []

        # Split the train/valid sets according to OmniSeg3D
    count = 900// image_stride + (1 if 900 % image_stride > 0 else 0)

    # train_ids = np.arange(0, 900, image_stride)
    # test_ids = np.array([x+image_stride//2 for x in train_ids])
    # print(train_ids,test_ids)
    # all_ids=np.concatenate((train_ids,test_ids),axis=0)

    sam_i=0
    all_idx = np.arange(0, 900,image_stride)
    test_idx = np.arange(0, len(all_idx), 4) # 25% of the images are used for validation
    train_idx = np.setdiff1d(all_idx, all_idx[test_idx])
    #print(len(train_idx))

    for idx in np.arange(0,900, image_stride):
        rgb_path = os.path.join(input_folder , "rgb" , f"rgb_{idx}.png")
        gt_seg_path= os.path.join(input_folder , "semantic_instance",f"semantic_instance_{idx}.png")
       # print(idx,rgb_path)
        # raise()
        image=Image.open(rgb_path)
        image_name=rgb_path.split("/")[-1].split(".")[0]
        depth_path =os.path.join(input_folder , "depth" , f"depth_{idx}.png")# depth_paths[idx]
        depth = Image.open(depth_path)
        depth = (transf(depth)/1000)#.squeeze()
 
        pose = poses[idx]
        id_masks=None

        sam_i=0
        if sam_folder=='split' or sam_folder=='overlap':
            if sam_paths and idx in train_idx:
                sam_mask=np.load(sam_paths[sam_i])
                sam_i+=1
                id_masks=torch.from_numpy(sam_mask).cuda()
                id_list=torch.unique(id_masks,sorted=True).cuda()
                for j,id in enumerate(id_list):
                    if id == 0:
                        continue
                    id_masks[id_masks==id]=j
                id_masks=id_masks.cpu().numpy()
        gt_seg=None
        if os.path.exists(gt_seg_path):
            gt_seg=cv2.imread(gt_seg_path,cv2.IMREAD_UNCHANGED)


        # try:
        #     sam_mask=np.load(os.path.join(sam_split_path,f'{image_name}_split.npy'))
        #     id_masks=torch.from_numpy(sam_mask).cuda()
        #     id_list=torch.unique(id_masks,sorted=True).cuda()
        #     for j,id in enumerate(id_list):
        #         if id == 0:
        #             continue
        #         id_masks[id_masks==id]=j
        #     id_masks=id_masks.cpu().numpy()
        # except:
        #     pass

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
            depth=depth 
        )
        cam_list.append(cam)
        # Split the train/valid sets according to OmniSeg3D
    all_idx = np.arange(0, len(cam_list))
    test_idx = np.arange(0, len(cam_list), 4) # 25% of the images are used for validation
    train_idx = np.setdiff1d(all_idx, test_idx)

    # train_camera_infos = cam_list[:count]
    # test_camera_infos = cam_list[count:]

    #if eval:
    train_camera_infos = [cam_list[idx] for idx in train_idx]
    test_camera_infos = [cam_list[idx] for idx in test_idx]
    # else:
    #     train_camera_infos = [cam_list[idx] for idx in train_idx]
    #     test_camera_infos = []
    
    # print("Train:", [cam.image_name for cam in train_camera_infos])
    # print()
    # print("Valid:", [cam.image_name for cam in valid_camera_infos])

    nerf_normalization = getNerfppNorm(train_camera_infos)

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_camera_infos,
        test_cameras=test_camera_infos,
        nerf_normalization=nerf_normalization,
        ply_path=str(ply_path)
        # scene_type=SceneType.REPLICA_SEMANTIC,
        # valid_mask_by_name={},
        # vignette_by_name={},
    )


    return scene_info