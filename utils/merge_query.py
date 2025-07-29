import torch
import torch.nn.functional as F
def cal_ious(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = (gt & pred).sum(dim=(-2, -1))
    union = (gt | pred).sum(dim=(-2, -1))
    return intersection / union.clamp(min=1)


def cal_accs(pred, gt):
    pred = pred > 0
    gt = gt > 0
    intersection = (gt & pred).sum(dim=(-2, -1))
    accs = intersection / pred.sum(dim=(-2, -1)).clamp(min=1)
    return accs

def mask_scores(gt, remains_mask, pred, acc_t=0.9):
    acc_whole = cal_accs(pred, gt)
    # acc_remains = cal_accs(pred, remains_mask)
    iou_remains = cal_ious(pred, remains_mask)
    iou_whole = cal_ious(pred, gt)
    acc = acc_whole > acc_t
    final_score = (iou_remains + iou_whole)*acc
    # final_score =  acc_whole*2+iou_whole*0.5#acc_whole + iou_whole*0.5

    return final_score

def get_query_pairs(gt, features, sample_num=256, acc_t=0.9):
    device = features.device
    cur_mask = torch.zeros_like(gt, dtype=torch.bool) 
    thresholds = torch.linspace(0, 1.3, steps=256, device=device).view(-1, 1, 1)
    
    querys, ths = [], []
    iou, n = 0, 0

    yx = torch.where(gt) 
    num_points = yx[0].shape[0]
    while iou < 0.99:
        remains_mask = gt & (~cur_mask)
        yx = torch.where(remains_mask)
        num_points = yx[0].shape[0]
        sample_num = min(sample_num, num_points)
        sampled_indices = torch.linspace(0, num_points - 1, steps=sample_num).long()
        sampled_y, sampled_x = yx[0][sampled_indices], yx[1][sampled_indices]
        if n >= sample_num:
            break
        y, x = sampled_y[n], sampled_x[n]
        q_pos = features[y, x]
        pred = get_pred(features, q_pos.unsqueeze(0), thresholds)
        scores= mask_scores(gt, remains_mask, pred, acc_t)
        best_idx = scores.argmax()
        th, pred = thresholds[best_idx], pred[best_idx]
        tmp_iou = cal_ious(cur_mask|pred, gt)
        n += 1
        if tmp_iou - iou >= 0.005: 
            querys.append(q_pos)
            ths.append(th)
            cur_mask=cur_mask|pred
            iou = tmp_iou
            n = 0
    return (torch.stack(querys), torch.stack(ths)) if querys else (None, None)

def get_pred(features,querys,ths):
    w, h, d = features.shape
    features_flat = features.view(-1, d)
    similar_pos = torch.cdist(features_flat,querys)
    similar_pos = similar_pos.view(w, h, -1)
    similar_pos = similar_pos.permute(2, 0, 1)
    pred = similar_pos < ths

    return pred