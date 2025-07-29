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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
from gaussian_renderer.origin import render
from gaussian_renderer.trace import trace
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.render_utils import save_img_f32, save_img_u8
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_obj_by_mask(gaussian,mask,inverse=False):
    if inverse:
        mask=~mask
    gaus=gaussian.capture('Objects',mask)
    obj=GaussianModel(gaussian.max_sh_degree)
    obj.restore(gaus,mode='obj')
    return obj


def render_by_mask(gaussians,mask,path,viewpoints,pipe, background):
    with torch.no_grad():
        if mask!=None:
            obj=get_obj_by_mask(gaussians,mask)
        else:
            obj=gaussians
        for idx, viewpoint in enumerate(viewpoints):
            render_pkg = render(viewpoint, obj, pipe, background)
            img=render_pkg["render"]
            if img.sum()  < 200:
                continue      
            os.makedirs(path,exist_ok=True)
            save_img_u8(img.permute(1,2,0).cpu().numpy(),os.path.join(path,'RGB_{0:05d}'.format(idx) + ".png"))

def get_weights(gaussians, viewpoints, pipe, background, unseen=-1, alpha_w=False):
    with torch.no_grad():
        weights = torch.zeros((gaussians.get_opacity.shape[0], len(viewpoints)), dtype=torch.int).cuda()
        #print(len(viewpoints))
        for idx, view in enumerate(viewpoints):
            sam_mask = view.sam_mask.copy()  # 创建连续内存的副本 torch.tensor([1,2]).cuda()
            id_masks = torch.tensor(sam_mask, dtype=torch.int16, device="cpu")
                # 然后再移至GPU
            id_masks = id_masks.cuda()
            id_masks[id_masks > 1] = 0
            w = trace(view, gaussians, id_masks, id_masks.max(), pipe, background, alpha_w=alpha_w)
            unseen_mask = (w.sum(-1) == 0)
            w = torch.argmax(w, dim=-1)
            w[unseen_mask] = -1 
            weights[:,idx] = w   
    return weights

def split_mask(gaussians, viewpoints, pipe, background, threshold=2, sp_th=1, soft_th=0.8, alpha_w=False):
    with torch.no_grad():
        nums = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.int16).cuda()
        ab_nums = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.int16).cuda()
        for idx, view in enumerate(viewpoints):
            # 同样修复内存错误
            sam_mask = view.sam_mask.copy()  # 创建连续内存的副本 torch.tensor([1,2]).cuda()
            id_masks = torch.tensor(sam_mask, dtype=torch.int16, device="cpu")
            id_masks = id_masks.cuda()
            
            w = trace(view, gaussians, id_masks, id_masks.max(), pipe, background, alpha_w)
            seen = w.sum(-1) > threshold
            value, _ = torch.max(w, dim=-1)
            value = value / (w.sum(-1) + 1e-6)
            ab = (value < soft_th) & seen
            nums += seen
            ab_nums += ab

        sp_mask = (ab_nums / (nums + 1e-6)) > sp_th

    return sp_mask


def prune_mask(gaussians,viewpoints,pipe, background,unseen=-1,alpha_w=False):
    weights = get_weights(gaussians,viewpoints,pipe, background,unseen,alpha_w)
    p_mask = ((weights!=unseen).sum(-1)==0)
    return p_mask.cuda()

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint,alpha_w):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=-1)
    gaussians.training_setup(opt)
    first_iter = 0

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    ob_ratio=0

    pcycle = opt.split_cycle_num
    cycle_from = opt.split_from_iter
    cycle_interval = opt.split_cycle_interval
    scale = 0.8#越大分出来的越小
    threshold = 25#unseen th
    soft_th = 0.8 #越小越不严格 0.8
    # sp_th = 0.5 #越大越不严格 0.5
    sp_th = 0.4
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    result_file=os.path.join(dataset.model_path,'split_result.txt')
    with open(result_file,'a') as f:
        f.write(f'\nsp_th {sp_th} cycle interval {opt.split_cycle_interval}, cycle_num {opt.split_cycle_num}, depth {opt.gt_depth}\n')
        f.write(f'Prune {opt.prune} \n')
    viewpoints = scene.getTrainCameras().copy()
    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii, depth= render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg['surf_depth']

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # regularization
        lambda_normal = opt.lambda_normal 
        lambda_dist = opt.lambda_dist 

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss 
        
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
            ema_depth_for_log = ob_ratio


            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "depth": f"{ema_depth_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if opt.prune:
                if iteration == 1 or iteration== opt.iterations:
                    #len(viewpoints)
                    with torch.no_grad():
                        p_mask = prune_mask(gaussians,viewpoints,pipe, background,unseen=-1).cuda()
                        gaussians.prune_points(p_mask)
                        print('delete {} gaussians, total {} gaussians now'.format(p_mask.sum(),gaussians.get_opacity.shape[0]))
                        with open(result_file,'a') as f:
                            f.write('delete {} gaussians, total {} gaussians now\n'.format(p_mask.sum(),gaussians.get_opacity.shape[0]))

            if opt.split and iteration >= opt.split_from_iter:
                if pcycle > 0 and (iteration - cycle_from)%cycle_interval==0:
                    sp_mask = split_mask(gaussians,viewpoints,pipe, background,threshold=threshold,sp_th=sp_th,soft_th=soft_th,alpha_w=alpha_w)#获取要分裂的gaus
                    pre_split_num=sp_mask.sum()
                    print(f'sp_mask num:{pre_split_num} total:{gaussians.get_opacity.shape[0]}, ratio {pre_split_num/gaussians.get_opacity.shape[0]}, mean scale {gaussians.get_scaling[sp_mask].mean()}')
                    if pcycle == opt.split_cycle_num:
                        with open(result_file,'a') as f:
                            f.write(f'Before: sp_mask num:{pre_split_num} total:{gaussians.get_opacity.shape[0]}, ratio {pre_split_num/gaussians.get_opacity.shape[0]}, mean scale {gaussians.get_scaling[sp_mask].mean()}\n')

                    gaussians.weights_split(sp_mask, opt.opacity_cull, scale=scale)
                    ob_ratio = sp_mask.sum()
                    sp_mask = split_mask(gaussians,viewpoints,pipe, background,threshold=threshold,sp_th=sp_th,soft_th=soft_th,alpha_w=alpha_w)#删除仍是ab的gaus
                    gaussians.weights_prune(sp_mask, 0, -1)
                    pcycle -= 1
       
            if  iteration == opt.iterations:
                sp_mask=split_mask(gaussians,viewpoints,pipe,background,threshold=threshold,sp_th=sp_th,soft_th=soft_th,alpha_w=alpha_w)
                print(sp_mask.sum())
                with open(result_file,'a') as f:
                    f.write(f'After: sp_mask num:{pre_split_num} total:{gaussians.get_opacity.shape[0]}, ratio {pre_split_num/gaussians.get_opacity.shape[0]}, mean scale {gaussians.get_scaling[sp_mask].mean()}\n')
            if iteration % 1000 == 0:
                #Current ab gaussians
                sp_mask = split_mask(gaussians,viewpoints,pipe,background,threshold=threshold,sp_th=sp_th,soft_th=soft_th,alpha_w=alpha_w)
                print(sp_mask.sum(),gaussians.get_scaling[sp_mask].mean())

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(f"{dataset.sam_folder}_{iteration}")

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + f"{dataset.sam_folder}" + ".pth")

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),result_file)

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,result_file):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                with open(result_file,'a') as f:
                    f.write("[ITER {}] Evaluating {}: L1 {} PSNR {}\n".format(iteration, config['name'], l1_test, psnr_test))
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6010)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1, 3000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--alpha_w", action="store_true", help="True for alpha_w")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.test_iterations, args.checkpoint_iterations,\
                 args.start_checkpoint, args.alpha_w)

    # All done
    print("\nTraining complete.")