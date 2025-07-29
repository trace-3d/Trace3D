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
import sys

from random import randint
from utils.loss_utils import contrastive_loss
from gaussian_renderer import  network_gui
from gaussian_renderer.render_feature import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams

def training_feature(dataset, opt, pipe, save_iterations, checkpoint, save_name):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    assert checkpoint is not None, "Checkpoint path must be provided to load a model."
    
    (model_params, first_iter) = torch.load(checkpoint)
    first_iter = 0
    gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    contrastive_func = contrastive_loss

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
        
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, include_feature=True)
        instance_image = render_pkg["instance_image"]
        instance_features = instance_image.permute(1, 2, 0).view(-1, 16)
        
        temperature = 100
        main_loss = 0

        sam_mask = torch.from_numpy(viewpoint_cam.sam_mask).clone().to(torch.int).view(-1).cuda()
        sample_num = opt.sample_num
        n_sample = min(int(len(sam_mask)/sample_num)+1, 1)

        #filter out the zero instances
        instance_features = instance_features[sam_mask > 0]
        sam_mask = sam_mask[sam_mask > 0]

        index = torch.randperm(len(instance_features)).cuda()
        for sample_i in range(n_sample):
            sample_idx = index[sample_i*sample_num:(sample_i+1)*sample_num]
            features = instance_features[sample_idx]
            instance_labels = sam_mask[sample_idx]
            con_loss = contrastive_func(features, instance_labels, temperature)
            main_loss += con_loss
        loss = main_loss / n_sample

        total_loss = loss
        total_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()
                
            torch.cuda.empty_cache()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            if (iteration in save_iterations):
                save_path=os.path.join(scene.model_path, dataset.sam_folder, 'chkpnt')
                os.makedirs(save_path, exist_ok=True)
                save_path= os.path.join(save_path, save_name + str(iteration) + ".pth") #TODO distinguish load chkpnt by load_iter
                torch.save((gaussians.capture('Features'), iteration),save_path)
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
 

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Instance feature training parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--save_name", type=str,help="name of stored checkpoint")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training_feature(lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations,  args.start_checkpoint,args.save_name)

    # All done
    print("\nTraining complete.")
