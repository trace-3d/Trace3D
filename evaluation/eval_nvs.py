import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from utils.merge_query import get_query_pairs, get_pred, cal_accs, cal_ious
from utils.metric_utils import labels_and_depths, get_ref_view, get_view_ids, overlay_prediction, ex_ref_view
from datetime import datetime
from utils.render_utils import save_img_u8
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer.render_feature import render
import numpy as np
from tqdm import tqdm
import torch

test_ids = {
    'office_0': [3, 4, 7, 8, 9, 10, 12, 14, 17, 19, 21, 23, 26, 28, 29, 30, 36, 37, 40, 42, 44, 46, 54, 55, 57, 58, 61],
    'office_1': [3, 7, 9, 11, 13, 14, 15, 17, 23, 24, 29, 32, 33, 36, 37, 39, 42, 44, 45, 46],
    'office_2': [2, 8, 9, 13, 14, 17, 19, 23, 27, 40, 41, 47, 49, 51, 54, 58, 60, 65, 67, 70, 71, 72, 73, 78, 85, 90, 92, 93],
    'office_3': [3, 8, 11, 14, 15, 18, 19, 25, 29, 30, 32, 33, 38, 39, 43, 51, 54, 55, 61, 65, 72, 76, 78, 82, 87, 91, 95, 96, 101, 111],
    'office_4': [1, 2, 6, 7, 9, 11, 17, 22, 23, 26, 33, 34, 39, 47, 49, 51, 52, 53, 55, 56],
    'room_0': [5, 6, 7, 10, 13, 14, 16, 25, 32, 33, 35, 46, 51, 53, 55, 60, 64, 67, 68, 83, 86, 87, 92],
    'room_1': [1, 2, 4, 6, 7, 9, 10, 11, 16, 18, 24, 28, 32, 36, 37, 44, 48, 52, 54, 56],
    'room_2': [3, 5, 6, 7, 8, 9, 11, 12, 16, 18, 22, 26, 27, 37, 38, 39, 40, 43, 49, 55, 56]
}

def feature_map(viewpoint_cam, gaussians, pipe, background):
    with torch.no_grad():
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, include_feature=True)
        render_features = render_pkg["instance_image"].permute(1, 2, 0)
    return render_features

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--iter", default=30000, type=int)
    parser.add_argument("--start_checkpoint", default=None, type=str)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument("--skip_feat_norm", action="store_true")
    args = get_combined_args(parser)
    assert args.save_path != None

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    (model_params, first_iter) = torch.load(args.start_checkpoint)
    gaussians.restore(model_params, mode='render')
    method = dataset.sam_folder

    scene_name = os.path.basename(dataset.source_path)
    train_viewpoints = scene.getTrainCameras().copy()
    test_viewpoints = scene.getTestCameras().copy()

    ids = torch.tensor(test_ids[scene_name], dtype=torch.uint8)
    ids = torch.unique(ids)

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    EXP_SAVE_PATH = args.save_path
    save_path = os.path.join(EXP_SAVE_PATH, f'{scene_name}')
    rgb_save_path = os.path.join(EXP_SAVE_PATH, f'{scene_name}', 'rgb')
    ref_save_path = os.path.join(EXP_SAVE_PATH, f'{scene_name}', 'ref')
    overview_path = os.path.join(EXP_SAVE_PATH, 'overview')
    os.makedirs(overview_path, exist_ok=True)
    os.makedirs(ref_save_path, exist_ok=True)
    out_dir = os.path.join(args.save_path, f'{scene_name}/{time}/{method}')

    tbar_outer, ious_all = tqdm(enumerate(ids), 'Objects', N_OBJ := ids.numel()), torch.zeros(N_OBJ)
    accs_all = torch.zeros(N_OBJ)
    all_viewpoints = train_viewpoints+test_viewpoints
    reject_view = []
    if scene_name == 'office_1':
        reject_view = np.arange(474, 504)
        print(f"Rejecting views for office_1: {reject_view}")
    elif scene_name == 'office_4':
        reject_view = np.arange(618, 734)
        print(f"Rejecting views for office_4: {reject_view}")

    if len(reject_view) > 0:
        original_count = len(all_viewpoints)
        all_viewpoints = [view for view in all_viewpoints if int(
            view.image_name[4:]) not in reject_view]  # "rgb_{idx}"
        print(f"Filtered viewpoints from {original_count} to {len(all_viewpoints)}")

    sams = []
    for viewpoint in all_viewpoints:
        sams.append(feature_map(viewpoint, gaussians, pipe, background).to('cpu'))
    sams = torch.stack(sams)
    view_ids = get_view_ids(all_viewpoints)
    distance_type = 'euclidean'
    sample_num = 96
    acc_t = 0.9  # acc_threshold for each patch

    if distance_type == 'cosine':
        sams = torch.nn.functional.normalize(sams, dim=3)

    for index, ref_id in tbar_outer:
        os.makedirs(os.path.join(rgb_save_path, f'{ref_id}'), exist_ok=True)
        labels, depths = labels_and_depths(all_viewpoints)  # labels:gt_masks of all objects
        gt = (labels == ref_id).cuda()
        probab = 1

        ref_view, all_views = get_ref_view(labels, depths, ref_id)
        ex_ref_view = ex_ref_view(scene_name, ref_id)

        if ex_ref_view is not None:
            ex_ref_view_id = view_ids.index(ex_ref_view)
            assert ex_ref_view_id in all_views
            ref_view = ex_ref_view_id

        tbar_inner, probab = tqdm(enumerate(all_views), 'Frames',
                                  N_VIEWS := all_views.numel(), False), 1
        if (N_VIEWS == 0):
            continue
        ious = torch.zeros(N_VIEWS)
        accs = torch.zeros(N_VIEWS)

        querys, ths = get_query_pairs(gt[ref_view], sams[ref_view].cuda(), sample_num, acc_t=acc_t)
        k = 1
        while querys is None:
            querys, ths = get_query_pairs(
                gt[ref_view], sams[ref_view].cuda(), sample_num, acc_t=acc_t-k*0.1)
            k += 1
        save_cnt = 0
        for n, view in tbar_inner:
            pred = get_pred(sams[view].cuda(), querys, ths)
            pred = torch.any(pred, dim=0)
            ious[n] = cal_ious(pred, gt[view])
            accs[n] = cal_accs(pred, gt[view])
            tbar_inner.set_postfix(iou_frame=ious[n].item(
            ), iou_object=ious[:n + 1].mean().item(), num=len(querys))
            if view == ref_view or save_cnt % 10 == 0:
                image = all_viewpoints[view].original_image.permute(1, 2, 0).clone()
                vis = overlay_prediction(image, pred, gt[view])
                if view == ref_view:
                    save_img_u8(vis.cpu().numpy(), os.path.join(
                        ref_save_path, f'{ref_id}_{all_viewpoints[view].image_name}_{100 * ious[n].item():.1f}.png'))
                else:
                    save_img_u8(vis.cpu().numpy(), os.path.join(
                        rgb_save_path, f'{ref_id}', f'{all_viewpoints[view].image_name}_{100 * ious[n].item():.1f}.png'))
            if save_cnt % 80 == 0:
                save_img_u8(vis.cpu().numpy(), os.path.join(overview_path,
                            f'{scene_name}_{ref_id:03d}_{100 * ious[n].item():.1f}.png'))
            save_cnt += 1
        ious_all[index] = (iou := ious[1:].mean().nan_to_num_(0))
        accs_all[index] = (acc := accs[1:].mean().nan_to_num_(0))
        tbar_outer.set_postfix(iou_object=iou.item(), acc_o=acc.item(
        ), iou_scene=ious_all[:index + 1].mean().item(), acc_s=accs_all[:index + 1].mean().item())

    result_file = os.path.join(save_path, f'results_{time}.txt')
    np.savetxt(result_file, torch.stack((ids, 100 * ious_all), 1).cpu().numpy(), ('%d', '%.2f'))
    open(result_file, 'a').write(f'\nmIoU {100 * ious_all.mean():.2f}\n, acc_t {acc_t}')
