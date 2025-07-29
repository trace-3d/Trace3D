import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import random

from conf.con_masks_conf import *
import colorsys
GAUS_NUM_TH = 100
BG_ID = 0


def colormap(id_map):
    device = id_map.device
    num_ids = int(id_map.max().item()) + 1
    colors = []
    for i in range(num_ids):
        hue = (i * 0.618033988749895) % 1
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(rgb)
    colors = torch.tensor(colors, device=device)
    return colors[id_map.long().squeeze()]


def get_weight(classes, weight):
    c_w_masks = []
    count = []
    for c in classes:
        mask = (weight == c)
        count.append(mask.sum())
        if mask.sum() > GAUS_NUM_TH:
            c_w_masks.append(mask)
    count = torch.Tensor(count)
    return c_w_masks, count


def valid_weight_mask(w, unseen=-1, bg_id=BG_ID):
    return (w != unseen) & (w != bg_id)


def filter_weights(weights, ct_th, max_num, unseen=-1):

    uni, count = torch.unique(weights, return_inverse=False, return_counts=True, dim=0)
    count = count*(valid_weight_mask(uni).sum(-1) > ct_th+1)
    sorted_indices = torch.argsort(count, descending=True)
    uni = uni[sorted_indices[:max_num]]
    count = count[sorted_indices[:max_num]]
    return uni, count


def _get_miou(w_a: torch.Tensor,
              w_b: torch.Tensor,
              ct_th: int,
              max_view: int,
              unseen: int = -1) -> float:

    max_gaus_num = MAX_POINTS // w_a.shape[1]
    uni_a, count_a = filter_weights(w_a, ct_th, max_gaus_num, unseen)
    uni_b, count_b = filter_weights(w_b, ct_th, max_gaus_num, unseen)

    w_ab = count_a.unsqueeze(0) * count_b.unsqueeze(1)
    w_ab = w_ab.unsqueeze(-1)

    union = ((valid_weight_mask(uni_a)).unsqueeze(0)) & ((valid_weight_mask(uni_b)).unsqueeze(1))
    inter = ((uni_a).unsqueeze(0) == (uni_b).unsqueeze(1)) & union

    union = (union * w_ab).sum()
    inter = (inter * w_ab).sum()

    return inter / (union + 1e-6)


def cal_miou(mask_a: torch.Tensor,
             mask_b: torch.Tensor,
             weights: torch.Tensor,
             view: int,
             unseen: int = -1,
             vf_th: float = 0.2,) -> float:

    wl = weights[:, 0:view]
    wr = weights[:, view+1:]
    weights = torch.cat((wl, wr), dim=-1)

    w_a = weights[mask_a]  # class in {gaus_num,view}
    w_b = weights[mask_b]

    # if len(w_a) < 100 or len(w_b) < 100:
    #     return 0

    w_a_valid = valid_weight_mask(w_a).sum(0)/w_a.shape[0]  # view: w_a !unseen,!bg views/ all_views
    w_b_valid = valid_weight_mask(w_b).sum(0)/w_b.shape[0]
    view_filter = (w_a_valid > vf_th) & (w_b_valid > vf_th)
    val_views = view_filter.sum()

    if val_views < 2:
        return 0

    w_a = w_a[:, view_filter]
    w_b = w_b[:, view_filter]

    miou = 0
    n = 0
    max_num = MAX_VIEWS
    iter_num = val_views//max_num+1
    while n < iter_num:
        vals = torch.arange(n, val_views-1, iter_num, dtype=torch.int)
        # print(vals,val_views)
        miou += _get_miou(w_a[:, vals], w_b[:, vals], iter_num, max_num)
        n += 1
    return miou/iter_num


def process_mask(mask):
    id_list = torch.unique(mask, sorted=True).cuda()
    patch_mask = torch.zeros(mask.shape).cuda()
    for j, id in enumerate(id_list):
        patch_mask[mask == id] = j
    return patch_mask


class SegmentationMask:
    def __init__(self, sam_masks, view, image_name):
        self.view = view
        self.image_name = image_name
        self.repaired_num = 0
        self.rp_iou = []
        self.unseen = -1
        self.h, self.w = sam_masks.shape[-2:]
        self.mask, self.conflict = self._get_mask(sam_masks)
        self.heri = self.conflict.clone()

        self.origin_mask = self.mask.clone()
        self.repair_area = torch.zeros_like(self.mask, dtype=torch.int16)

        self.mask = process_mask(self.mask).to(torch.int16)
        self.pre_union_mask = self.mask.clone()

        self.c_area = self._get_c_area(sam_masks)
        self._pre_union()

    def _get_mask(self, sam_masks):
        conflict = torch.zeros((self.h, self.w)).cuda()

        if len(sam_masks.shape) == 2:
            return sam_masks, conflict

        elif len(sam_masks.shape) == 3:
            for m in sam_masks:
                conflict = conflict + (m > 0)
            N, H, W = sam_masks.shape
            flat_mask = sam_masks.permute(1, 2, 0).reshape(-1, N)
            _, inverse_indices = torch.unique(flat_mask, return_inverse=True, dim=0)
            patch_mask = inverse_indices.view(H, W)
            return patch_mask, conflict
        else:
            raise ValueError(f'Error masks: {sam_masks.shape}')

    def repair(self, u_weights, max_view=180, miou_th=0.5, curr_weight=None):
        self.c_class = self._get_c_class()
        if len(self.c_class) == 0:
            self.repaired_mask = self.mask.clone()
            return None
        pairs = []
        if curr_weight == None:
            curr_weight = u_weights[:, self.view]

        for cla in self.c_class:  # cla:confilict area
            c_w_masks, count = get_weight(cla, curr_weight)
            filter_mask = count > GAUS_NUM_TH
            cla = cla[filter_mask]

            while len(cla) > 1:
                for j in range(1, len(cla)):
                    miou = cal_miou(c_w_masks[0], c_w_masks[j], u_weights, self.view, self.unseen)
                    if miou > miou_th:
                        self.rp_iou.append(miou)
                        self.repaired_num += 1
                        pairs.append((cla[0].item(), cla[j].item()))
                        self._change_class_id(cla[0], cla[j])
                        c_w_masks[j] = c_w_masks[j] & c_w_masks[0]
                        # count[j]=count[j]+count[0]
                        break
                cla = cla[1:]
                c_w_masks = c_w_masks[1:]
                # count=count[1:]
            # for i in merge_area:

        self.repaired_mask = self.mask.clone()

        return pairs

    def _change_class_id(self, before, after):
        assert before != 0, 'Changing the background is illegal'
        if len(self.mask == before) == 0:
            return
        before_area = (self.mask == before)
        self.mask[before_area] = after
        self.repair_area[before_area] = after

    def _set_class_zero(self, classes):
        assert 0 not in classes, 'Changing the background is illegal'
        if len(classes) > 0:
            for c in classes:
                self.mask[self.mask == c] = 0

    def pre_process(self, sam_masks=None, th=1e-2):
        if len(sam_masks.shape) == 2:
            return

        for a in self.c_area:  # a confict area
            # Extract unique values and their counts (pixel counts) for the current mask area
            id_list, num = torch.unique(self.mask[a], return_counts=True)
            # Remove background (id == 0)
            valid_ids = id_list[id_list != 0]
            valid_counts = num[id_list != 0]
            val_mask = valid_counts / valid_counts.sum() < th
            max_area_id = valid_ids[valid_counts.argmax()]
            for region_id in valid_ids[val_mask]:
                region_mask = (self.mask == region_id)
                ious = ((region_mask & sam_masks).sum((-2, -1)) /
                        (region_mask | sam_masks).sum((-2, -1)).clamp(1)).max()
                if ious > 0.5:
                    continue
                self._change_class_id(region_id, max_area_id)

    def _get_c_class(self):
        c_class = []
        if len(self.c_area) == 0:
            return []
        for a in self.c_area:
            # Extract unique values and their counts (pixel counts) for the current mask area
            id_list, num = torch.unique(self.mask[a], return_counts=True)
            # Remove background (id == 0)
            valid_ids = id_list[id_list != 0]
            if len(valid_ids) > 1:
                c_class.append(valid_ids)
        return c_class

    def _get_mask_list_by_point(self, sam_masks, point):
        return sam_masks[:, point[0], point[1]] > 0

    def _union_area(self, sam_masks, mask_list):
        temp = torch.zeros((self.h, self.w)).cuda()
        for m in sam_masks[mask_list]:
            temp = temp+(m > 0)
        union_area = (temp > 0)
        inter_area = (temp > 1)
        iou = inter_area.sum()/union_area.sum()
        if iou < 0.05:
            self.conflict[inter_area] = self.conflict[inter_area] - temp[inter_area]
            return None
        self.conflict = self.conflict-temp
        return union_area

    def _find_conflict(self, sam_masks, point):

        mask_list = self._get_mask_list_by_point(sam_masks, point)

        area = self._union_area(sam_masks, mask_list)

        if area is not None:

            while self.conflict[area].max() > 0:

                con = self.conflict * area
                point = torch.where(con == con.max())
                point = (point[0][0].item(), point[1][0].item())

                mask_list = self._get_mask_list_by_point(sam_masks, point)
                new_area = self._union_area(sam_masks, mask_list)

                if new_area is not None:
                    area = area | new_area

        return area

    def _get_c_area(self, sam_masks):

        c_area = []

        while self.conflict.max() > 1:

            point = torch.where(self.conflict == self.conflict.max())
            point = (point[0][0].item(), point[1][0].item())

            area = self._find_conflict(sam_masks, point)

            if area is not None:

                main_type = torch.unique(self.mask[area], return_counts=True)[1].max()

                if main_type / area.sum() < 0.95:
                    c_area.append(area)
                else:

                    id = torch.unique(self.mask[area])
                    id = id[id != 0]
                    self.mask[area] = id[0]

        return c_area

    def show_area(self):
        if len(self.c_area) > 0:
            c_area = self.c_area
        else:
            return None
        con = torch.zeros((self.h, self.w)).cuda()
        k = 1
        for i in c_area:
            con += i*k
            k += 1
        alpha = (con != 0)
        img = colormap(con)
        img = torch.cat((img, alpha.unsqueeze(-1)), dim=-1)

        return img

    def show_mask(self, mask=None):
        img = self.mask
        if mask != None:
            img = mask
        alpha = (img != 0)
        img = colormap(img)
        img = torch.cat((img, alpha.unsqueeze(-1)), dim=-1)
        return img

    def visual(self, mask, skip_process=False):
        if not skip_process:
            mask = process_mask(mask)
        alpha = (mask != 0)
        mask = colormap(mask)
        mask = torch.cat((mask, alpha.unsqueeze(-1)), dim=-1)
        return mask

    def _pre_union(self):
        for area in self.c_area:
            id = torch.unique(self.pre_union_mask[area])[0]
            self.pre_union_mask[area] = id
        self.pre_union_mask = process_mask(self.pre_union_mask)

    def compare_mask(self) -> torch.Tensor:

        if self.repaired_mask is None:
            raise ValueError('empty masks, please repair first')

        def create_padding(height: int, width: int) -> torch.Tensor:
            return self.visual(torch.zeros((height, width)).cuda())

        visualizations = {
            'top_row': [
                self.visual(self.origin_mask),
                self._visualize_conflicts(),
            ],
            'bottom_row': [
                self.visual(self.repaired_mask),
                self.visual(self.repair_area),
            ]
        }

        h_padding = create_padding(self.h, 4)

        def combine_row(images: List[torch.Tensor]) -> torch.Tensor:

            images = [img if img.shape[-1] == 4 else torch.cat([img, torch.ones_like(img[..., :1])], dim=-1)
                      for img in images]
            return torch.cat([
                img if i == 0 else
                torch.cat([h_padding, img], dim=1)
                for i, img in enumerate(images)
            ], dim=1)

        top_row = combine_row(visualizations['top_row'])
        bottom_row = combine_row(visualizations['bottom_row'])

        v_padding = create_padding(4, top_row.shape[1])

        return torch.cat([top_row, v_padding, bottom_row], dim=0)

    def _visualize_conflicts(self) -> torch.Tensor:

        if not self.c_area:
            return self.visual(torch.zeros_like(self.origin_mask))

        conflict_mask = torch.zeros((self.h, self.w)).cuda()
        for idx, area in enumerate(self.c_area, start=1):
            conflict_mask += area * idx
        alpha = (conflict_mask != 0)
        conflict_mask = colormap(conflict_mask)
        conflict_mask = torch.cat((conflict_mask, alpha.unsqueeze(-1)), dim=-1)

        return conflict_mask
