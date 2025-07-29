#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch
from glob import glob
import cv2

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
   
    image_path: str
    image_name: str
    width: int
    height: int
    
    instance_image:np.array = None  #gt instance mask
    sam_mask:np.array = None
    depth:np.array = None #gt depth
    features:np.array = None #sam features, for faster online inference, only used in replica

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    # query_2d_seg: dict
    # seg_ids: torch.tensor

def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readCameras(scale_mats, world_mats, image_paths,instance_paths):
    def load_K_Rt_from_P(filename, P=None):
        if P is None:
            lines = open(filename).read().splitlines()
            if len(lines) == 4:
                lines = lines[1:]
            lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
            P = np.asarray(lines).astype(np.float32).squeeze()

        out = cv2.decomposeProjectionMatrix(P)
        K = out[0]
        R = out[1]
        t = out[2]

        K = K/K[2,2]
        intrinsics = np.eye(4)
        intrinsics[:3, :3] = K

        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = R.transpose()
        pose[:3,3] = (t[:3] / t[3])[:,0]

        return intrinsics, pose
    cam_infos = []
    for idx, mats in enumerate(zip(scale_mats, world_mats)):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(image_paths)))
        sys.stdout.flush()
        scale_mat, world_mat=mats
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        fx, fy = intrinsics[0, 0], intrinsics[0, 0]
        R=pose[:3, :3]
        T=-pose[:3,3]
        FovY = focal2fov(fy, 384)
        FovX = focal2fov(fx, 384)#TODO replace with height and width
        image_path=image_paths[idx]
        image_name = os.path.basename(image_path).split(".")[0]
        image=Image.open(image_path)
        if instance_paths:
            instance_path=instance_paths[idx]
            instance_image=Image.open(instance_path)
        else:
            instance_image=None
        cam_info = CameraInfo(uid=0, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=384, height=384,instance_image=instance_image)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def read_test_from_Traj(path, trajfile):
    scale = 384 / 680
    offset = (384 - 680 ) * 0.5
    inv_metrix = np.eye(4)
    inv_metrix[:3,1] *= -1
    inv_metrix[:3,2] *= -1
    cam_infos = []
    query_2dseg_path= Path(os.path.join(path, "2dseg_query.json"))
    query_json=None
    if query_2dseg_path.exists():
        print("Loading the 2D segmentation info")
        query_json = json.loads(open(query_2dseg_path, "r").read())

    image_paths = glob_data(os.path.join(path,'test_rgb', "*rgb.png"))
    instance_paths = glob_data(os.path.join(path,"test_segs", "*segs.png"))
    with open(os.path.join(path, trajfile)) as txt_file:
        lines = txt_file.readlines()

    idx=0
    for count,line in enumerate(lines):
        line = line.split() 
        if (count-10) % 80 != 0:
                continue
        c2w = np.array(list(map(float, line))).reshape(4, 4)
        R_w_c = c2w[:3, :3]
        T_c_w = - np.transpose(R_w_c) @ c2w[:3, 3]
        R_w_c[1, :] *= -1  
        R_w_c[2, :] *= -1
    
        replica_rgb_image_name = image_paths[idx]
        if os.path.exists(replica_rgb_image_name):
            rgb_name = replica_rgb_image_name
            focal_length_x = 600.0*scale
            focal_length_y = 600.0*scale

        image_name = f'test_rbg/{Path(rgb_name).stem}'
        query=None
        if query_json is not None:
            query=query_json[image_name]

        rgb = Image.open(rgb_name)

        FovX = focal2fov(focal_length_x, rgb.size[0])
        FovY = focal2fov(focal_length_y, rgb.size[1])
        if instance_paths:
            instance_image=cv2.imread(instance_paths[idx],cv2.IMREAD_UNCHANGED).astype(np.uint8)
        else:
            instance_image=None
        id_masks=None
        depth=None
        cam_infos.append(CameraInfo(uid=idx, R=R_w_c, T=T_c_w, FovY=FovY, FovX=FovX, image=rgb,
                                    image_path=rgb_name, image_name=image_name, width=rgb.size[0],
                                    height=rgb.size[1],
                                    instance_image=instance_image,
                                    sam_mask=id_masks,
                                    depth=depth,
                                    query=query))
        idx+=1

    return cam_infos

def readCamerasFromKeyFrameTraj(path, trajfile,sam_folder):
    scale = 384 / 680
    offset = (384 - 680 ) * 0.5
    inv_metrix = np.eye(4)
    inv_metrix[:3,1] *= -1
    inv_metrix[:3,2] *= -1
    train_cam_infos = []

    #instance_paths=None
    query_2dseg_path= Path(os.path.join(path, "2dseg_query.json"))
    query_json=None
    if query_2dseg_path.exists():
        print("Loading the 2D segmentation info")
        query_json = json.loads(open(query_2dseg_path, "r").read())
    image_paths = glob_data(os.path.join('{0}'.format(path),"*_rgb.png"))

    #image_paths = glob_data(os.path.join('{0}'.format(path),'images',"*_rgb.png"))
    ori=False
    instance_paths = glob_data(os.path.join('{0}'.format(path),"segs", "*segs.png"))
    if sam_folder in ['overlap','split','origin_overlap','pre_union','sam_origin_cover']:
        sam_paths = glob_data(os.path.join(path , "gd_sam" ,sam_folder ,"*.npy"))
    elif sam_folder in ['origin']:
        sam_paths= glob_data(os.path.join(path , "gd_sam" ,sam_folder ,"*.npy"))
        ori=True
    else:
        sam_paths=None
    #sam_paths = glob_data(os.path.join('{0}'.format(path),"gd_sam", "*mask.npy"))
    #sam_paths = glob_data(os.path.join('{0}'.format(path),"sam_only", "*mask.npy"))
    #sam_paths = glob_data(os.path.join('{0}'.format(path),"sam_masks_new/pre_process", "*pre.npy"))
    #depth_paths = glob_data(os.path.join('{0}'.format(path),"*depth.npy")) #gt_depth
    depth_paths = glob_data(os.path.join('{0}'.format(path),"*depthfm_depth", "*.npy"))

    with open(os.path.join(path, trajfile)) as txt_file:
        lines = txt_file.readlines()

    idx = 0
    count=-1
    for line in lines:
        line = line.split() 
        count += 1
        if count % 20 != 0:
                continue
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}".format(idx+1))
        sys.stdout.flush()

        c2w = np.array(list(map(float, line))).reshape(4, 4)
        

        R_w_c = c2w[:3, :3]
        T_c_w = - np.transpose(R_w_c) @ c2w[:3, 3]
        R_w_c[1, :] *= -1  
        R_w_c[2, :] *= -1
    
        replica_rgb_image_name = image_paths[idx]
        if os.path.exists(replica_rgb_image_name):
            rgb_name = replica_rgb_image_name
            focal_length_x = 600.0*scale
            focal_length_y = 600.0*scale

        image_name = Path(rgb_name).stem
        query =None
        if query_json is not None:
            query=query_json[image_name]

        rgb = Image.open(rgb_name)
        # d =Image.fromarray((np.array(Image.open(d_name)) / 6553.5).astype(np.float32), mode='F')

        FovX = focal2fov(focal_length_x, rgb.size[0])
        FovY = focal2fov(focal_length_y, rgb.size[1])
        if instance_paths:
            instance_image=cv2.imread(instance_paths[idx],cv2.IMREAD_UNCHANGED).astype(np.uint8)
        else:
            instance_image=None

        if sam_paths is not None:
            sam_mask=np.load(sam_paths[idx])
            id_masks=torch.from_numpy(sam_mask).cuda()
            if  ori and  len(id_masks.shape)==3:
                con_mask= torch.zeros((rgb.size[0],rgb.size[1])).cuda()
                k=len(id_masks)
                for i, m in enumerate(id_masks):
                    con_mask = con_mask + m*((i+k)*i+1)
                id_masks=con_mask  
            id_list=torch.unique(id_masks,sorted=True).cuda()
            for j,id in enumerate(id_list):
                if id == 0:
                    continue
                id_masks[id_masks==id]=j
                
            id_masks=id_masks.cpu().numpy()               
        else:
            id_masks=None
        
        if depth_paths:
            depth=np.load(depth_paths[idx])
        else:
            depth=None

        train_cam_infos.append(CameraInfo(uid=idx, R=R_w_c, T=T_c_w, FovY=FovY, FovX=FovX, image=rgb,
                                    image_path=rgb_name, image_name=image_name, width=rgb.size[0],
                                    height=rgb.size[1],
                                    instance_image=instance_image,
                                    sam_mask=id_masks,
                                    depth=depth,
                                    query=query))
        idx+=1
    sys.stdout.write('\n')

    return train_cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, dataset,sam_folder='non',test_idx=[],images='images'):
    cam_infos = []
    if sam_folder in ['overlap','split','origin_overlap','pre_union','sam_origin_cover']:
        sam_paths = glob_data(os.path.join(dataset , "sam" ,sam_folder ,"*.npy"))
    else:
        sam_paths=None
    sam_i=0
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length = intr.params[0]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(dataset, images,os.path.basename(extr.name))
        #print(image_path)
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        if (sam_paths is not None )and( idx not in test_idx ):
            sam_mask=np.load(sam_paths[sam_i])
            sam_i+=1
            id_masks=torch.from_numpy(sam_mask).cuda()
            id_list=torch.unique(id_masks,sorted=True).cuda()
            for j,id in enumerate(id_list):
                if id == 0:
                    continue
                id_masks[id_masks==id]=j
            id_masks=id_masks.cpu().numpy()
        else:
            id_masks=None



        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,sam_mask=id_masks)
        cam_infos.append(cam_info)
    sys.stdout.write(f'read {sam_i} sam masks')
    sys.stdout.write('\n')
    return cam_infos



def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    os.makedirs(os.path.dirname(path),exist_ok=True)
    ply_data.write(path)


def readReplicaSceneInfo_traj_feature(path, trajfile, eval, sam_folder):

    if trajfile==None:
        trajfile='traj.txt'
    train_cam_infos_unsorted = readCamerasFromKeyFrameTraj(path,trajfile,sam_folder)
    train_cam_infos = sorted( train_cam_infos_unsorted .copy(), key = lambda x : x.image_name)
    train_cam_infos = train_cam_infos

    if eval:
        test_cam_infos_unsorted = read_test_from_Traj(path,trajfile)
        test_cam_infos = sorted( test_cam_infos_unsorted .copy(), key = lambda x : x.image_name)
        test_cam_infos = test_cam_infos
    else:
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
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


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSceneInfo(path, read_instance, eval, llffhold=8):
    def glob_data(data_dir):
        data_paths = []
        data_paths.extend(glob(data_dir))
        data_paths = sorted(data_paths)
        return data_paths
    image_paths = glob_data(os.path.join('{0}'.format(path),"images", "*_rgb.png"))
    instance_paths = None
    if read_instance:
        instance_paths = glob_data(os.path.join('{0}'.format(path), "images","instance_mask", "*.png"))
    cameras = os.path.join(path,"cameras.npz")
    n_images = len(image_paths)    
    
    camera_dict = np.load(cameras)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    cam_infos_unsorted = readCameras(scale_mats,world_mats,image_paths,instance_paths)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    #nerf_normalization = None

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readColmapSceneInfo(path, images, eval,sam_folder,llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    # cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir),sam_folder=sam_folder)
    # cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if 'llff' in path: # LLFF data
        test_idx = 1 if 'fern' in path \
            else 8 if 'horns' in path \
            else 13 if 'orchids' in path \
            else 31 if 'trex' in path \
            else 0 # consistent with NVOS
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, \
                                               dataset=path,sam_folder=sam_folder,test_idx=[test_idx],images=reading_dir)
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        # seg_path=glob(os.path.join(path,"*mask.png"))
        # assert len(seg_path)==1, f"empty or muti mask, {seg_path}"
        # seg_name=os.path.basename(seg_path[0])[:-9]
        

        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx  != test_idx]
        #assert cam_infos[test_idx].image_name == seg_name, f"{seg_name}"
        # resized_image_PIL = cv2.imread(seg_path[0]).resize(resolution)
        # resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
        #cam_infos[test_idx] = cam_infos[test_idx]._replace(instance_image=Image.open(seg_path[0]))
        #print(cv2.imread(seg_path[0]).sum())
        test_cam_infos = [cam_infos[test_idx]]
        # if split=='train':
        #     img_paths.pop(test_idx)
        #     self.poses = np.delete(self.poses, test_idx, 0)
    else:
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, dataset=path,sam_folder=sam_folder,images=reading_dir)
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        if eval:
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        else:
            train_cam_infos = cam_infos
            test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            storePly(ply_path, xyz, rgb)
        except:
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

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
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

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Replica": readReplicaSceneInfo_traj_feature
}