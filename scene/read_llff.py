import os
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
from PIL import Image
import torch
import sys
from scene.dataset_readers import SceneInfo, getNerfppNorm, storePly, fetchPly, glob_data, CameraInfo, read_points3D_binary
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

DEFAULT_SAM_FOLDER = 'sam'

def readColmapCameras(cam_extrinsics, cam_intrinsics, dataset, sam_folder=None, test_idx=[], images='images'):
    cam_infos = []
    sam_paths = None
    if sam_folder is not None:
        sam_paths = glob_data(os.path.join(dataset, DEFAULT_SAM_FOLDER, sam_folder, "*.npy"))

    sam_i = 0
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

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
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

        image_path = os.path.join(dataset, images, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        
        id_masks = None
        if len(sam_paths) > 0 and (idx not in test_idx):
            sam_mask = np.load(sam_paths[sam_i])
            sam_i += 1
            if len(sam_mask.shape) == 3:
                N, H, W = sam_mask.shape
                sam_mask = torch.from_numpy(sam_mask).cuda()
                flat_mask = sam_mask.permute(1, 2, 0).reshape(-1, N)
                _, inverse_indices = torch.unique(flat_mask, return_inverse=True, dim = 0)
                id_masks = inverse_indices.view(H, W).cpu().numpy()
            else:
                id_masks = sam_mask

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, sam_mask=id_masks)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def read_llff_data(path, eval, sam_folder):
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

    images = "images"

    test_idx = 1 if 'fern' in path \
        else 8 if 'horns' in path \
        else 13 if 'orchids' in path \
        else 31 if 'trex' in path \
        else 0  
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           dataset=path, sam_folder=sam_folder, test_idx=[test_idx], images=images)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx != test_idx]
    test_cam_infos = [cam_infos[test_idx]]

    nerf_normalization = getNerfppNorm(train_cam_infos)

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
