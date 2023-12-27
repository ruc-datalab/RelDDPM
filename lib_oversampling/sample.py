import sys
sys.path.append('../')
import os
from ddpm import modules, diffusion, train
import torch
import data_utils as du
from ddpm.resample import create_named_schedule_sampler

def get_cond_fn(controller, scale_factor, label, n_classes=2):
    
    def cond_fn(c, x, t):
        x = x.float()
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = controller(x_in, t)
            if n_classes == 2:
                if label == 1.0:
                    gradients = torch.autograd.grad(logits.sum(), x_in)[0] * scale_factor
                elif label == 0.0:
                    gradients = -torch.autograd.grad(logits.sum(), x_in)[0] * scale_factor
            return gradients

    return cond_fn


def oversampling(n_samples, controller, diffuser, label, device, n_classes=2, scale_factor=8.0):
    controller.to(device)
    diffuser.to(device)
    diffuser.variables_to_device(device)

    cond_fn = get_cond_fn(controller, scale_factor, label, n_classes)
    cond = torch.zeros(n_samples, 1)

    samples = diffuser.sample(n_samples, control_tools=[cond, cond_fn])
    return samples
