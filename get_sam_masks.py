# from groundingdino.util.inference import load_model, load_image, predict, annotate
# from groundingdino.util import box_ops
from segment_anything import build_sam, SamAutomaticMaskGenerator
import numpy as np
import matplotlib.pyplot as plt
import torch
import os, sys
from cv2 import imread
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min()+1e-6)
    map = (map * 255).round().long().squeeze()
    map = colors[map]
    return map

def visual_masks(masks):
    h, w = masks.shape[-2:]
    total_mask= torch.zeros((h, w)).cuda()
    if len(masks.shape)==2:
        total_mask=masks
    else:
        k=len(masks)
        for i, m in enumerate(masks):
            total_mask = total_mask + m*((i+k)*i+1) 
    alpha=(total_mask!=0)
    img = colormap(total_mask)
    img = torch.cat((img,alpha.unsqueeze(-1)),dim=-1)
    return img

    
def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

if __name__ == "__main__":
    parser = ArgumentParser(description="SAM segmentation")
    parser.add_argument('--sam_checkpoint', type=str, default='/mnt/fillipo/shenhongyu/model/SAM/sam_vit_h_4b8939.pth')
    parser.add_argument('--file_path', type=str, help='Path to the images folder.')
    args = parser.parse_args()
    
    print("Initializing SAM...")
    sam_checkpoint = args.sam_checkpoint
    sam = build_sam(checkpoint=sam_checkpoint).to(device=DEVICE)

    # mask_generator = SamAutomaticMaskGenerator(sam)
    mask_generator = SamAutomaticMaskGenerator(
        sam,
        stability_score_thresh=.9,

    )#for llff
    
    save_folder = 'origin'
    SAM = 'sam'
    save_path = os.path.join(os.path.dirname(args.file_path), SAM, save_folder)
    os.makedirs(save_path, exist_ok=True)
    image_paths = glob_data(os.path.join(args.file_path, "*.*"))
    
    print('Load {} images.'.format(len(image_paths)))
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path).split('.')[0]
        image = imread(image_path)
        results = mask_generator.generate(image)
        masks = []
        for r in results:
            masks.append(r['segmentation'].astype(bool))
        masks = np.array(masks)
        for m, mask in enumerate(masks):
            union = (mask & masks[m + 1:]).sum((1, 2), True)
            masks[m + 1:] |= mask & (union > .9 * mask.sum((0, 1), True))
        np.save(os.path.join(save_path, f'{image_name}.npy'), masks)
        masks = torch.from_numpy(masks).cuda()
        visual_mask = visual_masks(masks)
        plt.imsave(os.path.join(save_path, f'{image_name}_sam.png'), visual_mask.cpu().numpy())

           