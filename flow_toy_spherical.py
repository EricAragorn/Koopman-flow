import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import sample_2d_data

from modules.kernel_fn import *
from modules.latent_sampler import *
from modules.preimage import *
from dataset import ADNIMRDataset

from modules.utils import cartesian_to_unit_sphere_3d, unit_sphere_to_cartesian_3d, normalize

device = 'cuda:1'

def scale(data, target_range, margin=1e-5):
    scales = []
    offsets = []
    with torch.no_grad():
        for i in range(data.size(1)):
            column_max, column_min = data[:, i].max(), data[:, i].min()
            scale = (target_range[i][1] - target_range[i][0]) / (column_max - column_min) - 2 * margin
            offset = margin - target_range[i][0] - column_min * scale
            data[:, i] = data[:, i] * scale + offset
            scales.append(scale)
            offsets.append(offset)
    return data, scales, offsets

def unscale(data, scales, offsets):
    with torch.no_grad():
        for i in range(data.size(1)):
            data[:, i] = (data[:, i] - offsets[i]) / scales[i]
    return data

def main(args):

    fig_data = plt.figure()
    fig_data.set_size_inches(15, 15)
    ax_data = fig_data.add_subplot(111)
    fig_data.suptitle("Data")
    fig_samples = plt.figure()
    fig_samples.set_size_inches(15, 15)
    ax_samples = fig_samples.add_subplot(111)
    fig_samples.suptitle("Sampled")
    fig_dist = plt.figure()
    fig_dist.set_size_inches(15, 15)
    ax_dist = fig_dist.add_subplot(111)
    fig_dist.suptitle("Distribution")

    data_type_list = ["8gaussians", "2spirals", "checkerboard", "rings"]

    for data_t in data_type_list:
        data_t_subdir = os.path.join(args.sample_dir, data_t)
        if not os.path.exists(data_t_subdir):
            os.mkdir(data_t_subdir)

        data, scales, offsets = scale(sample_2d_data(data_t, args.pf_sample_size).to(device), [[0., np.pi], [0., 2 * np.pi]])
        data = cartesian_to_unit_sphere_3d(data)
        input_kernel = NeuralKernel(3, 20000, n_layers=4, kernel_type='ntk', activation='relu')
        output_kernel = MixRBFKernel(3, sigma=[0.15])

        all_data = unscale(unit_sphere_to_cartesian_3d(data), scales, offsets).detach().cpu().numpy()
        ax_data.clear()
        ax_data.set_xlim(-4, 4)
        ax_data.set_ylim(-4, 4)
        ax_data.scatter(all_data[:, 0], all_data[:, 1])

        data_file = os.path.join(args.sample_dir, "data_{:s}.png".format(data_t))
        fig_data.savefig(data_file)

        prior_sample_fn = lambda size: normalize(torch.randn(size, 3), p=2)

        eps_list = [1e-6]
        
        def update(eps):
            compression = False
            model = KernelPFGenerator(input_kernel=input_kernel,
                                      output_kernel=output_kernel,
                                      output_samples=[data], 
                                      labels=None, 
                                      preimage_module=GeodesicInterpPreimage(),
                                      prior_sample_fn=prior_sample_fn, 
                                      nystrom_compression=compression, 
                                      epsilon=eps, 
                                      nystrom_points=1000, 
                                      p_dim=-1,
                                      device=device).to(device)
            reference_samples = normalize(torch.randn(10000, 3), p=2, dim=-1).to(device)
            density_op = model.get_density_operator(reference_samples)

            N = M = 100
            eval_x = torch.linspace(0., np.pi, N)
            eval_y = torch.linspace(0., 2 * np.pi, M)
            eval_coords = torch.stack([eval_x.repeat_interleave(eval_y.size(0)), eval_y.repeat(eval_x.size(0))], dim=-1).to(device)
            eval_coords_sphere = cartesian_to_unit_sphere_3d(eval_coords)

            eval_coords = unscale(eval_coords, scales, offsets).view(N, M, 2)

            p = density_op(eval_coords_sphere).view(N, M)
            ax_dist.clear()
            ax_dist.contourf(eval_coords[:, :, 0].detach().cpu().numpy(), 
                         eval_coords[:, :, 1].detach().cpu().numpy(), 
                         p.detach().cpu().numpy())
            dist_file = os.path.join(data_t_subdir, "distribution_{:s}_eps_{:.1e}.png".format(data_t, eps))
            fig_dist.savefig(dist_file)

            print("Generating samples")
            model.eval()
            samples = []
            for i in tqdm(range(100)):
                batch_samples, _ = model.sample(100, topk=5)
                samples.append(batch_samples[0])
            
            all_samples = unscale(unit_sphere_to_cartesian_3d(torch.cat(samples, dim=0)), scales, offsets).detach().cpu().numpy()
            ax_samples.clear()
            ax_samples.set_xlim(-4, 4)
            ax_samples.set_ylim(-4, 4)
            ax_samples.scatter(all_samples[:, 0], all_samples[:, 1])
            ax_samples.set_title("eps = {:e}".format(eps))

            sample_file = os.path.join(data_t_subdir, "samples_{:s}_eps_{:.1e}.png".format(data_t, eps))
            fig_samples.savefig(sample_file)
        for eps in eps_list:
            update(eps)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths and names
    parser.add_argument("--experiment_id", default="toy_spherical", type=str)
    parser.add_argument("--sample_dir", default="./samples", type=str)

    parser.add_argument("--pf_sample_size", default=10000, type=int)
    args = parser.parse_args()

    args.sample_dir = os.path.join(args.sample_dir, args.experiment_id)
    
    for _dir in [args.sample_dir]:
        os.makedirs(_dir, exist_ok=True)

    main(args)