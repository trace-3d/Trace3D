import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.metric_utils import labels_and_depths, get_ref_view,get_view_ids, get_obj_by_mask, overlay_prediction
from utils.merge_query import get_query_pairs, get_pred, cal_ious, cal_accs
from utils.image_utils import psnr
from utils.render_utils import save_img_u8
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from gaussian_renderer.trace import trace
from gaussian_renderer.render_feature import render
from scene import Scene, GaussianModel
import torch
import json
import shutil
import numpy as np
from tqdm import tqdm

def feature_map(viewpoint_cam, gaussians, pipe, background):
    with torch.no_grad():
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, include_feature=True)
        render_features = render_pkg["instance_image"].permute(1, 2, 0)
    return render_features

def get_bounding_box_mask(mask, pad=10):
    if not torch.any(mask):
        return torch.zeros_like(mask, dtype=torch.uint8)
    rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)
    y_indices, x_indices = torch.where(rows)[0], torch.where(cols)[0]

    H, W = mask.shape
    y_min, y_max = max(y_indices[0] - pad, 0), min(y_indices[-1] + pad, H - 1)
    x_min, x_max = max(x_indices[0] - pad, 0), min(x_indices[-1] + pad, W - 1)

    bbox_mask = torch.zeros_like(mask, dtype=torch.uint8)
    bbox_mask[y_min:y_max + 1, x_min:x_max + 1] = 1
    return bbox_mask

def cal_bbox_psnr(img, gt_img, gt_mask, pad=10):
    bbox_mask = get_bounding_box_mask(gt_mask, pad)
    test_gt = gt_img.clone()
    test_gt[:, (gt_mask == 0)] = 0
    ps = psnr(img[:, (bbox_mask == 1)], test_gt[:, (bbox_mask == 1)]).mean()
    return ps

def test_object(gaussians, gaus_mask, id, test_viewpoints, background, pipe, object_path, background_path):
    miou, mps, accs = [], [], []
    object, left = get_obj_by_mask(gaussians, gaus_mask), get_obj_by_mask(gaussians, ~gaus_mask)
    save_mark = 0
    for idx, viewpoint in enumerate(test_viewpoints):
        gt_img = viewpoint.original_image.clone()
        gt_id_masks = torch.from_numpy(viewpoint.original_instance_image).to(torch.int).cuda()
        gt_mask = (gt_id_masks == id)
        if gt_mask.sum() < 200:
            continue
        render_pkg = render(viewpoint, object, pipe, background, False)
        img = render_pkg["render"].clone()
        img = img.clip(0, 1)
        render_mask = img.sum(0) > 0
        iou = cal_ious(render_mask, gt_mask)
        acc = cal_accs(render_mask, gt_mask)
        ps = cal_bbox_psnr(img, gt_img, gt_mask)
        miou.append(iou)
        accs.append(acc)
        if not torch.isinf(ps):
            mps.append(ps)

        if save_mark % 2 == 0:
            left_pkg = render(viewpoint, left, pipe, background, False)
            img = render_pkg["render"].permute(1, 2, 0).cpu().numpy()
            vis_left = left_pkg["render"].permute(1, 2, 0).cpu().numpy()
            save_img_u8(img, os.path.join(
                object_path, f'{viewpoint.image_name}_{iou:.3f}_{acc:.3f}.png'))
            save_img_u8(vis_left, os.path.join(background_path,
                        f'{viewpoint.image_name}_bg_{iou:.3f}_{acc:.3f}.png'))
        save_mark += 1
    if len(miou) > 0:
        miou = torch.stack(miou).mean().cpu().numpy()
        mps = torch.stack(mps).mean().cpu().numpy()
        macc = torch.stack(accs).mean().cpu().numpy()
        return miou, macc, mps
    else:
        return 0, 0, 0

def creat_path(target_path):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
        print(f"Deleted existing folder: {target_path}")
        os.makedirs(target_path, exist_ok=False)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--start_checkpoint", default=None, type=str)
    parser.add_argument("--alpha_w", action="store_true", help="True for alpha_w")
    parser.add_argument('--save_path', type=str, default=None,help="Path to save visualization results")
    parser.add_argument('--result_save_path', type=str, default=None,help="Path to save result json")
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--clean_history', action="store_true")
    parser.add_argument("--save_gaus_mask", action="store_true")
    args = get_combined_args(parser)
    assert args.save_path != None
    assert args.result_save_path != None
    assert args.method != None
    if args.clean_history:
        print("---------------------------------------------------------\n")
        print("-----------------------clean history---------------------\n")
        print("---------------------------------------------------------\n")
    alpha_w = args.alpha_w
    model_name = os.path.basename(args.model_path)
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint = args.start_checkpoint
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, mode='render')
    else:
        raise ValueError("checkpoint missing!")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    scene_name = os.path.basename(dataset.source_path)
    # --------------------------------------------------------------------------------------------- init
    ids = torch.tensor({
        "office_0": [3, 4, 7, 9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 29, 30, 32, 34, 35, 36, 37, 40, 44, 48, 49, 57, 58, 61, 66],
        "office_1": [3, 8, 9, 11, 12, 13, 14, 17, 23, 24, 29, 30, 31, 32, 34, 35, 37, 43, 45,],
        "office_2": [0, 2, 3, 4, 6, 8, 9, 12, 13, 14, 17, 23, 27, 34, 38, 39, 46, 49, 51, 54, 57, 58, 59, 63, 65, 68, 69, 70, 72, 73, 74, 75, 77, 78, 80, 84, 85, 86, 90, 92, 93],
        "office_3": [1, 2, 8, 11, 12, 15, 18, 21, 22, 25, 29, 32, 33, 42, 51, 54, 55, 56, 60, 61, 70, 82, 85, 86, 88, 86, 97, 101, 102, 103, 110, 111,],
        "office_4": [3, 4, 5, 6, 9, 13, 16, 18, 20, 23, 31, 34, 47, 48, 49, 51, 52, 56, 60, 61, 62, 65, 69, 70, 71,],
        "room_0": [1, 2, 3, 4, 6, 7, 8, 11, 13, 15, 18, 19, 20, 21, 22, 24, 30, 32, 34, 35, 36, 39, 40, 41, 43, 45, 47, 49, 50, 51, 54, 55, 58, 61, 63, 64, 68, 69, 70, 71, 72, 73, 74, 75, 78, 79, 83, 85, 86, 87, 90, 92,],
        "room_1": [3, 4, 6, 7, 8, 9, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23, 24, 27, 30, 32, 33, 35, 37, 39, 40, 43, 45, 46, 48, 50, 51, 52, 53, 54,],
        "room_2": [2, 4, 5, 10, 14, 15, 17, 18, 19, 20, 22, 24, 26, 27, 28, 29, 31, 32, 34, 36, 38, 39, 40, 42, 44, 46, 47, 48, 49, 52, 54, 55, 56, 57, 58, 59, 61],
    }[scene_name], dtype=torch.uint8)

    ids = torch.unique(ids).int()

    viewpoints = scene.getTrainCameras().copy()
    test_viewpoints = scene.getTestCameras().copy()
    reject_view = []
    if scene_name == 'office_1':
        reject_view = np.arange(474, 504)
        print(f"Rejecting views for office_1: {reject_view}")
    elif scene_name == 'office_4':
        reject_view = np.arange(618, 734)
        print(f"Rejecting views for office_4: {reject_view}")

    if len(reject_view) > 0:
        viewpoints = [view for view in viewpoints if int(
            view.image_name[4:]) not in reject_view]  # "rgb_{idx}"
        test_viewpoints = [view for view in test_viewpoints if int(
            view.image_name[4:]) not in reject_view]  # "rgb_{idx}"

    view_ids = get_view_ids(viewpoints)
    sample_num = 96
    acc_t = 0.9 # acc_threshold for each patch
    trace_th = 0.66

    sams = []
    for viewpoint in viewpoints:
        sams.append(feature_map(viewpoint, gaussians, pipe, background).to('cpu'))
    sams = torch.stack(sams)

    test_sams = []
    for viewpoint in test_viewpoints:
        test_sams.append(feature_map(viewpoint, gaussians, pipe, background).to('cpu'))
    test_sams = torch.stack(test_sams)

    # ---------------------------------------------------------------load occ views
    file_path = './script/view_wo_occ_new.json'
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            view_data = json.load(f)
    view_data = view_data[scene_name]
    # ----------------------------------------------------------------------------------

    tbar_outer, ious_all = tqdm(enumerate(ids), 'Objects', N_OBJ := ids.numel()), torch.zeros(N_OBJ)
    with torch.no_grad():
        save_path = os.path.join(args.save_path, scene_name)
        if args.clean_history:
            creat_path(save_path)
        else:
            os.makedirs(save_path, exist_ok=True)
        method = f'{args.method}_trace_th_{trace_th}'
        if args.alpha_w:
            method = f"{method}_alpha_w"

        labels, depths = labels_and_depths(viewpoints)  # labels:gt_masks of all objects
        final_miou, final_accs, final_ps = [], [], []
        final_metrics = {
            "miou_3d": [],
            "ps": [],
            "accs": [],
            "miou_2d": [],
            "macc_2d": []
        }
        tested_ids = []
        result_save_path = args.result_save_path
        os.makedirs(os.path.basename(result_save_path), exist_ok=True)

   # -------------------------------------------------------------------------------------- load view
        for index, ref_id in tbar_outer:
            try:
                curr_test_views = view_data[str(ref_id.item())]
            except:
                curr_test_views = None
            object_path = os.path.join(save_path, 'objects', f'{ref_id}')
            background_path = os.path.join(save_path, 'background', f'{ref_id}')
            mask_rgb_path = os.path.join(save_path, 'mask', f'{ref_id}')
            os.makedirs(object_path, exist_ok=True)
            os.makedirs(background_path, exist_ok=True)
            os.makedirs(mask_rgb_path, exist_ok=True)

            tested_ids.append(ref_id.item())
            masks = []
            gt = (labels == ref_id).cuda()
            ref_view, all_views = get_ref_view(labels, depths, ref_id)

    # -------------------------------------------------------------------------------------get_muti-view consistent masks
            save_mark = 0
            mask_iou_2d, mask_acc_2d = [], []
            querys, ths = get_query_pairs(
                gt[ref_view], sams[ref_view].cuda(), sample_num, acc_t=acc_t)
    # -------------------------------------------------save pred on training view for trace
            for i in range(len(sams)):
                pred = get_pred(sams[i].cuda(), querys, ths)
                pred = torch.any(pred, dim=0)
                gt_mask = gt[i]
                if gt_mask.sum() >= 200:
                    mask_iou_2d.append(iou := cal_ious(pred, gt_mask))
                    mask_acc_2d.append(acc := cal_accs(pred, gt_mask))
                    if i == ref_view:
                        view = viewpoints[i]
                        gt_img = view.original_image
                        vis = overlay_prediction(gt_img.clone().permute(
                            1, 2, 0), pred, gt_mask).cpu().numpy()
                        save_img_u8(vis, os.path.join(
                            mask_rgb_path, f"ref_{viewpoints[i].image_name}_{iou.item():.2f}_{acc.item():.2f}.png"))
                    # if save_mark%16==0:
                    #     view = viewpoints[i]
                    #     gt_img = view.original_image
                    #     vis=overlay_prediction(gt_img.clone().permute(1, 2, 0),pred,gt_mask).cpu().numpy()
                    #     save_img_u8(vis,os.path.join(mask_rgb_path,f"{view.image_name}_{iou.item():.2f}_{acc.item():.2f}.png"))
                    save_mark += 1
                if pred.sum() < 200:
                    pred = pred | 0
                masks.append(pred)
    # -------------------------------------------------continue test on test-views, only for 2D
            for i in range(len(test_sams)):
                pred = get_pred(test_sams[i].cuda(), querys, ths)
                pred = torch.any(pred, dim=0)
                view = test_viewpoints[i]
                gt_masks = torch.from_numpy(view.original_instance_image).int().cuda()
                gt_mask = (gt_masks == ref_id)
                if gt_mask.sum() >= 200:
                    mask_iou_2d.append(iou := cal_ious(pred, gt_mask))
                    mask_acc_2d.append(acc := cal_accs(pred, gt_mask))
                    if save_mark % 16 == 0:
                        gt_img = view.original_image
                        vis = overlay_prediction(gt_img.clone().permute(
                            1, 2, 0), pred, gt_mask).cpu().numpy()
                        save_img_u8(vis, os.path.join(mask_rgb_path,
                                    f"{view.image_name}_{iou.item():.2f}_{acc.item():.2f}.png"))
                    save_mark += 1

            miou_2d = torch.tensor(mask_iou_2d).mean().cpu().numpy()
            macc_2d = torch.tensor(mask_acc_2d).mean().cpu().numpy()
    # --------------------------------------------------------------------------------------trace objects
            weights = []
            for idx, viewpoint in enumerate(viewpoints):
                w = trace(viewpoint, gaussians, masks[idx], 1, pipe, background, alpha_w)  # [P,2]
                weights.append(w)
            weights = torch.stack(weights)
            weights = weights.sum(0)  # [view,P,2]->[P,2]

            gaus_mask = weights[:, 1]/weights.sum(1).clamp(1e-9) > trace_th
    # --------------------------------------------------------------------------------------filter occ views
            if curr_test_views is not None:
                tmp_test_viewpoints = [view for view in test_viewpoints if int(
                    view.image_name[4:]) in curr_test_views]
            else:
                tmp_test_viewpoints = test_viewpoints.copy()
    # --------------------------------------------------------------------------------------save gaus mask
            gaus_mask_path = os.path.join(scene.model_path, "objects")
            if args.save_gaus_mask:
                os.makedirs(gaus_mask_path, exist_ok=True)
                torch.save(gaus_mask, os.path.join(gaus_mask_path, f"{ref_id}_gaus_mask.pt"))

    # --------------------------------------------------------------------------------------test objects
            miou, macc, mps = test_object(gaussians, gaus_mask, ref_id, tmp_test_viewpoints,
                                          background, pipe, object_path, background_path=background_path)
    # ---------------------------------------------------------------------------------------save results
            final_metrics["miou_3d"].append(miou)
            final_metrics["accs"].append(macc)
            final_metrics["ps"].append(mps)
            final_metrics["miou_2d"].append(miou_2d)
            final_metrics["macc_2d"].append(macc_2d)

        scene_avg = {
            "mIoU-object": float(np.mean(final_metrics["miou_3d"])),
            "mPrec-object": float(np.mean(final_metrics["accs"])),
            "PSNR-object": float(np.mean(final_metrics["ps"])),
            "mIoU-scene": float(np.mean(final_metrics["miou_2d"])),
            "mAcc-scene": float(np.mean(final_metrics["macc_2d"]))
        }

        mean_miou = scene_avg["mIoU-object"]
        mean_acc = scene_avg["mPrec-object"]
        mean_psnr = scene_avg["PSNR-object"]
        mean_miou_scene = scene_avg["mIoU-scene"]

        open('eval_3d.txt', 'a').write(
            f'\n {scene_name}: mIoU {mean_miou*100:.2f}  mAcc {mean_acc*100:.2f}  mPNSR {mean_psnr:.2f} Scene_IOU { mean_miou_scene} {checkpoint}')
        print(f'\n {scene_name}: mIoU {mean_miou*100:.2f}  mAcc {mean_acc*100:.2f}  mPNSR {mean_psnr:.2f} Scene_IOU { mean_miou_scene} {checkpoint}')
