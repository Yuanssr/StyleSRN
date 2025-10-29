import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import sys
from model.StyleSRN import showPIL,show_batch_images,show_batch_images_mask


def unpatchify(x, patch_size: int = 4, shape: tuple = (32, 128)):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    h = shape[0] // p
    w = shape[1] // p
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
    return imgs

import torchvision

def visualize_tensor(img_lr: torch.Tensor, img_hr: torch.Tensor, img_sr: torch.Tensor,
                     mask: torch.Tensor,refine_sr: torch.Tensor, patch_size: int = 4,H=32,W=128):
    """Construct visualization tensor for tensorboard

    Args:
        img_lr (torch.Tensor): Low resolution image tensor in shape (N, C, H/2, W/2)
        img_hr (torch.Tensor): High resolution image tensor in shape (N, C, H, W)
        img_sr (torch.Tensor): Super resolution image tensor in shape (N, C, H, W)
        reconstruction (torch.Tensor): Reconstructed image tensor in shape (N, C, H, W)
        mask (torch.Tensor): Mask tensor in shape (N, L)
        patch_size (int, optional): Patch size. Defaults to 4.

    Returns:
        torch.Tensor: Visualization tensor in shape (N, C, H, W)
    """

    imagenet_mean = torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    imagenet_std = torch.Tensor([0.5, 0.5, 0.5]).view(-1, 1, 1)
    torch.manual_seed(0)
    # randomly select 10 images for visualization
    idx = np.random.choice(img_sr.shape[0], 64, replace=False)

    img_lr = img_lr.detach().cpu()
    img_hr = img_hr.detach().cpu()
    img_sr = img_sr.detach().cpu()
    mask = mask.detach().cpu()
    refine_sr=refine_sr.detach().cpu()
    #plt.figure()
    #plt.imshow(confidence_map[0])
    #plt.axis('off')

    # mask
    mask = mask.unsqueeze(-1).repeat(1, 1, 48)
    mask = unpatchify(mask)
    # Masked image
    masked_img = img_sr * (1 - mask)

    img_lr = F.interpolate(img_lr, scale_factor=2, mode='bicubic', align_corners=True)

    #plt.figure()
    #plt.imshow(masked_img[:, :3, ...][30].permute(1, 2, 0).cpu())
    #plt.axis('off')
    # Concatenate images into a tensor
    tensor = torch.cat([img_lr, img_hr, img_sr, masked_img, refine_sr], dim=2)
    # Make a grid of images
    grid_img = torchvision.utils.make_grid(tensor, nrow=8, padding=2)

    return grid_img


def visualize_text_distribution(x):
    """

    Args:
        x: [N,T,C]

    Returns:

    """
    plt.figure()
    n = x.shape[0]
    for j in range(n):
        plt.subplot(2, 4, j+1)
        plt.imshow(x[j].cpu(),cmap='gray')
        plt.axis('off')


if __name__ == '__main__':
    img = torch.rand(10, 3, 32, 128)
    reconstruction = torch.rand(10, 256, 48)
    mask = torch.rand(10, 256)
    figures = visualize_tensor(img, reconstruction, mask)
    plt.show()
