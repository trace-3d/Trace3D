import torch

def overlay_mask_torch(image, mask, color=[235/255, 122/255, 119/255], alpha=0.5):
    """
    Image: H W C, cuda tensor
    Mask: H W, cuda tensor
    """
    image = image.clone()  # H W C
    if image.max() > 1:
        image /= 255.0
    mask = mask.bool()
    overlay = torch.zeros_like(image)
    color = torch.tensor(color).cuda()
    overlay[mask] = color
    combined = (1 - alpha) * image + alpha * overlay
    return combined.clamp(0, 1)
