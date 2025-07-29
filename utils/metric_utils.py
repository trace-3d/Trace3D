import torch
import numpy as np
from scene.gaussian_model import GaussianModel
import matplotlib.pyplot as plt

def get_view_ids(all_viewpoints):
    view_ids = []
    for view in all_viewpoints:
        view_id = int(view.image_name[4:])
        view_ids.append(view_id)
    return view_ids

def labels_and_depths(views):
    labels = []
    depths = []
    for view in views:
        label = view.original_instance_image
        labels.append(label)
        depths.append(view.gt_depth)
    return torch.tensor(np.stack(labels)).cuda(), torch.tensor(np.stack(depths)).cuda()

def get_ref_view(labels, depths, ref_id):
    gt = (labels == ref_id)
    areas=depths.clone().squeeze()
    areas[gt==0]=0
    areas_sum = areas.sum(dim=(1, 2)) 
    views = areas_sum.argsort(descending=True)
    ref_view, all_views = views[0], views[:(areas_sum != 0).sum()]
    
    return ref_view, all_views

def colormap(map, cmap="turbo"):
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    map = (map - map.min()) / (map.max() - map.min()+1e-6)
    map = (map * 255).round().long().squeeze()
    map = colors[map]
    return map

def visual(mask):
    alpha=(mask!=0)
    mask = colormap(mask)
    mask = torch.cat((mask,alpha.unsqueeze(-1)),dim=-1)
    return mask

def overlay_mask_on_image(image, mask, mask_color=(1, 0, 0), alpha=0.5):
    image = image / 255.0 if image.max() > 1 else image
    mask = mask.squeeze()
    overlay_image = image.copy() * (1 - alpha)
    overlay_image[mask == 1] = overlay_image[mask == 1]  + np.array(mask_color) * alpha
    return overlay_image

def overlay_prediction(image, pred_mask, gt_mask, alpha=0.5):
    image = image.clone()  
    if image.max() > 1: 
        image /= 255.0
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    tp_color = torch.tensor([1.0, 1.0, 0]).cuda() 
    fp_color = torch.tensor([1.0, 0, 0]).cuda()  
    fn_color = torch.tensor([0, 1.0, 0]).cuda() 

    overlay = torch.zeros_like(image)

    tp_region = pred_mask & gt_mask  # True Positive
    fp_region = pred_mask & ~gt_mask  # False Positive
    fn_region = ~pred_mask & gt_mask  # False Negative

    overlay[tp_region] = tp_color
    overlay[fp_region] = fp_color
    overlay[fn_region] = fn_color

    combined = (1 - alpha) * image + alpha * overlay
    return combined.clamp(0, 1) 

def get_obj_by_mask(gaussian, mask, inverse=False):
    if inverse:
        mask = ~mask
    gaus = gaussian.capture('Objects', mask)
    obj = GaussianModel(gaussian.max_sh_degree)
    obj.restore(gaus, mode='obj')
    return obj

ex_ref = {
    "office_0": {8: 185, 10: 138, 26: 815, 28: 635, 36: 634, 40: 177, 58: 646, 61: 203},
    "office_1": {7: 234, 33: 215, 17: 30, 39: 208, 45: 623, 46: 656},
    "office_2": {9: 592, 41: 585, },
    "office_3": {3: 307, 11: 321, 38: 2, 101: 156, },
    "office_4": {1: 28, 2: 146, 6: 493, 17: 12, 26: 138, 39: 801, 7: 569, 53: 35, },
    "room_0": {25: 108, 35: 868, 53: 247, 46: 844},
    "room_1": {1: 180, 6: 453, 10: 174, 36: 352, 48: 450, 56: 144, },
    "room_2": {3: 5, 7: 116, 9: 337, 11: 37, 16: 446, 18: 177, }
}


def ex_ref_view(scene_name: str, ref_id: int):
    return ex_ref.get(scene_name, {}).get(int(ref_id))