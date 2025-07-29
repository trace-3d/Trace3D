from pathlib import Path
from typing import List, Tuple, Optional
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from gaussian_renderer.render_feature import render
import os
from utils.con_mask_utils import SegmentationMask  
from utils.render_utils import save_img_u8
from gaussian_renderer.trace import trace
from scene import Scene, GaussianModel
from conf.con_masks_conf import *  
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

class MaskRepairPipeline:

    
    def __init__(self, dataset_path: str, scene: Scene, gaussians: GaussianModel):
        self.dataset_path = Path(dataset_path)
        self.scene_path=dataset_path
        self.scene = scene
        self.gaussians = gaussians

        self.camera_stack = scene.getTrainCameras().copy()
        self.dataset=None
        self.setup_directories()
        self.train_idx = None
        if 'llff' in dataset_path:
            self.dataset='llff'
            self.sam_name=self.nvos_sam
        elif 'replica' in dataset_path:
            self.sam_name = self.replica_sam
            self.dataset='replica'
        
    def setup_directories(self) -> None:
        sam_path = self.dataset_path / DEFAULT_SAM_FOLDER
        for folder in [SPLIT_FOLDER, COMPARE_FOLDER]:
            (sam_path / folder).mkdir(parents=True, exist_ok=True)
            
    def replica_sam(self,image_name):
        # image_id=int(image_name[4:])
        # return f"{image_id:05d}_masks_sam.npy"
        return f"{image_name}.npy"
    
    def nvos_sam(self,image_name):
        return f"{image_name}.npy"
    
    def get_train_indices(self, sam_paths: List[Path]) -> np.ndarray:

        if self.dataset=='replica':
            all_idx = np.arange(900)
            test_idx = np.arange(0, 900, 4)
            train_idx=np.setdiff1d(all_idx, test_idx)
            if self.scene == 'office_1':
                reject_idx = np.arange(474, 504) 
                return np.setdiff1d(train_idx,reject_idx)  
            elif self.scene  == 'office_4':
                reject_idx = np.arange(618, 734)
                return np.setdiff1d(train_idx,reject_idx)
        elif self.dataset=='llff':
            test_idx = 1 if 'fern' in self.scene_path  \
            else 8 if 'horns' in self.scene_path \
            else 13 if 'orchids' in self.scene_path \
            else 31 if 'trex' in self.scene_path \
            else 0 
            train_idx=np.arange(len(sam_paths))
            print(train_idx)
            return np.setdiff1d(train_idx,test_idx)
        else:
            train_idx=np.arange(len(sam_paths))
        return train_idx

        
    def compute_weights(self, 
                       masks: List[SegmentationMask],
                       camera_stack: List,
                       pipe,
                       background: torch.Tensor,
                       alpha_w:bool=False,
                       mask_type:str='mask') -> torch.Tensor:
        weights = torch.zeros((self.gaussians.get_opacity.shape[0], 
                             len(camera_stack))).cuda()#[p,view]
        
        for idx, mask in enumerate(masks):
            if mask_type=='union':
                p_mask = mask.pre_union_mask.to(torch.int)
            else:
                p_mask = mask.mask.to(torch.int)
            view = camera_stack[mask.view]
            w = trace(view, self.gaussians, p_mask, p_mask.max(), pipe, background, alpha_w)#[p,class]
            unseen = (w.sum(-1) == 0)
            w = torch.argmax(w, dim=-1)
            w[unseen] = UNSEEN_VALUE
            weights[:, idx] = w
            
        return weights
        
    def repair_masks(self, pipe, background: torch.Tensor, alpha_w:bool=False) -> None:
        sam_paths = list(self.dataset_path.glob(f"{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/*.npy"))
        self.train_idx = self.get_train_indices(sam_paths)
        masks = []
        for idx, camera in tqdm(enumerate(self.camera_stack)):
            if self.dataset=='replica':
                sam_data = np.load(f"{self.dataset_path}/{DEFAULT_SAM_FOLDER}/{ORIGIN_FOLDER}/{self.sam_name(camera.image_name)}")
            elif self.dataset=='llff':
                sam_data = np.load(sam_paths[self.train_idx[idx]])
            sam_tensor = torch.from_numpy(sam_data).cuda().squeeze()
            sam_tensor = sam_tensor[sam_tensor.sum((-2, -1)) > 96].squeeze()
            mask = SegmentationMask(sam_tensor, view=idx, image_name= camera.image_name)
            # mask.pre_process(sam_tensor, 0.001)
            masks.append(mask)

        for iteration in range(1):
            self._repair_iteration(masks, iteration,pipe,background)

    def _repair_iteration(self, 
                         masks: List[SegmentationMask],
                         iteration,
                         pipe, 
                         background: torch.Tensor, 
                         alpha_w:bool=False ,
                         ) -> None:
        num = 0
        sam_path = self.dataset_path / DEFAULT_SAM_FOLDER
        weights = self.compute_weights(masks, self.camera_stack, pipe, background, alpha_w)
        os.makedirs(sam_path / COMPARE_FOLDER/f'iter={iteration}',exist_ok=True)
        for i, mask in tqdm(enumerate(masks), total=len(masks)):
            mask.repair(weights, miou_th=0.4)
            num+=mask.repaired_num
            
            if len(mask.rp_iou) > 0:
                miou = torch.stack(mask.rp_iou).mean().item()
            else:
                miou = 0
            plt.imsave(
                sam_path / COMPARE_FOLDER/f'iter={iteration}' / '{}_{}_{:.2f}.png'.format(mask.image_name,mask.repaired_num,miou),
                mask.compare_mask().cpu().numpy()  
            )
            
            np.save(
                sam_path / SPLIT_FOLDER / f'{mask.image_name}.npy',
                mask.mask.cpu().numpy()
            )
            

def main():

    parser = ArgumentParser(description="merge patches")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--start_checkpoint", type=str, default="")
    parser.add_argument("--skip_pre", action="store_false")
    parser.add_argument("--include_feature", action="store_false")
    parser.add_argument("--interval", type=int, default=-1)
    parser.add_argument("--alpha_w", action="store_true")
    args = get_combined_args(parser)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    dataset.sam_folder = "empty" #prevent sam loading
    scene = Scene(dataset, gaussians, shuffle=False, load_iteration=-1)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pipeline = MaskRepairPipeline(args.source_path, scene, gaussians)
    pipeline.repair_masks(pipe, background, args.alpha_w)


if __name__ == "__main__":
    main()
