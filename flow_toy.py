import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import time
import os
from tqdm import tqdm
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import sample_2d_data

from modules.kernel_fn import *
from modules.flow import *
from modules.latent_sampler import *
from modules.preimage import *
from modules.utils import mix_rbf_mmd
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
    sns.color_palette("rocket", as_cmap=True)

    fig_data = plt.figure()
    fig_data.set_size_inches(15, 15)
    ax_data = fig_data.add_subplot(111)

    fig_data_dist = plt.figure()
    fig_data_dist.set_size_inches(15, 15)
    ax_data_dist = fig_data_dist.add_subplot(111)
    fig_samples = plt.figure()
    fig_samples.set_size_inches(15, 15)
    ax_samples = fig_samples.add_subplot(111)
    fig_dist = plt.figure()
    fig_dist.set_size_inches(15, 15)
    ax_dist = fig_dist.add_subplot(111)

    data_type_list = ["8gaussians", "2spirals", "checkerboard", "rings"]

    for data_t in data_type_list:
        gc.collect()
        torch.cuda.empty_cache()

        data_t_subdir = args.sample_dir

        data = sample_2d_data(data_t, args.pf_sample_size).to(device)

        all_data = data
        ax_data.clear()
        ax_data.set_axis_off()
        ax_data.set_xlim(-4, 4)
        ax_data.set_ylim(-4, 4)
        ax_data.scatter(all_data[:, 0].detach().cpu().numpy(), all_data[:, 1].detach().cpu().numpy())

        data_file = os.path.join(args.sample_dir, "data_{:s}.png".format(data_t))
        fig_data.savefig(data_file, bbox_inches='tight', pad_inches=0)

        ax_data_dist.clear()
        ax_data_dist.set_axis_off()
        ax_data_dist.hist2d(data[:, 0].detach().cpu().numpy(), 
                        data[:, 1].detach().cpu().numpy(),
                        bins=(200, 200),
                        cmap='magma',
                        range=[[-4, 4], [-4, 4]],
                        density=True)
        data_dist_file = os.path.join(data_t_subdir, "data_dist_{:s}_{:s}.png".format(data_t, args.experiment_id))
        fig_data_dist.savefig(data_dist_file, bbox_inches='tight', pad_inches=0)

        def save_density_fig(density_op):
            N = M = 200
            eval_x = torch.linspace(-4., 4, N)
            eval_y = torch.linspace(-4, 4, M)
            eval_coords = torch.stack([eval_x.repeat_interleave(eval_y.size(0)), eval_y.repeat(eval_x.size(0))], dim=-1).to(device)

            eval_coords_grid = eval_coords.view(N, M, 2)

            p_list = []
            p_batchsize = 5000
            with torch.no_grad():
                for i in range((N * M) // p_batchsize + int((N * M) % p_batchsize > 0)):
                    batch_p = density_op(eval_coords[(i * p_batchsize): ((i + 1) * p_batchsize)])
                    p_list.append(batch_p)
            p = torch.cat(p_list, dim=0).view(N, M)

            npy_file = os.path.join(data_t_subdir, "p_{:s}_{:s}.npy".format(data_t, args.experiment_id))
            np.save(npy_file, p.detach().cpu().numpy())
            
            ax_dist.clear()
            ax_dist.set_axis_off()
            ax_dist.contourf(eval_coords_grid[:, :, 0].detach().cpu().numpy(), 
                            eval_coords_grid[:, :, 1].detach().cpu().numpy(), 
                            p.detach().cpu().numpy(),
                            cmap='magma',
                            vmin=0.,
                            levels=250)
            dist_file = os.path.join(data_t_subdir, "distribution_{:s}_{:s}.png".format(data_t, args.experiment_id))
            fig_dist.savefig(dist_file, bbox_inches='tight', pad_inches=0)


        if args.density_estimator == "kpf":
            input_kernel = NeuralKernel(3, 10000, n_layers=4, kernel_type='ntk', activation='relu')
            output_kernel = MixRBFKernel(2, sigma=[0.4])
            prior_sample_fn = lambda size: torch.randn(size, 3)
            compression = False
            model = KernelPFGenerator(input_kernel=input_kernel,
                                        output_kernel=output_kernel,
                                        output_samples=[data], 
                                        labels=None, 
                                        # preimage_module=WeightedMeanPreimage(),
                                        preimage_module=MDSPreimage(),
                                        prior_sample_fn=prior_sample_fn, 
                                        nystrom_compression=compression, 
                                        epsilon=1e-5, 
                                        nystrom_points=1000, 
                                        p_dim=-1,
                                        device=device).to(device)
            reference_samples = torch.rand(10000, 2) * 8 - 4 # sample from reference density
            density_op = model.get_density_operator(reference_samples, epsilon=1e-3)
        elif args.density_estimator == "gmm":  
            model = GMMGenerator([data], n_components=10)
            def density_op(X):
                ret = torch.tensor(model.models[0].score_samples(X.detach().cpu().numpy())).exp()
                return ret
        elif args.density_estimator == "glow":
            model = Glow1d(2, 64, 50).to(device)

            def density_op(X):
                return model(X).exp()

            batchsize = 256
            datasize = args.pf_sample_size
            optimizer = torch.optim.Adam(model.parameters(), 1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2000, verbose=True)
            model(data[:256]) # init actnorm
            for i in range(500):
                index = np.random.permutation(datasize)
                _iter = tqdm(range(datasize // batchsize))
                for j in _iter:
                    batch_data = data[index[j * batchsize: (j + 1) * batchsize]]
                    logp = model(batch_data)
                    loss = -logp.mean()

                    _iter.set_postfix({"Bits per dim": loss.item()})

                    optimizer.zero_grad()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()

                    scheduler.step(loss)
                torch.save((i, model.state_dict()), "./cache/flow_toy_{:s}_{:s}.pth".format(args.experiment_id, data_t))
                save_density_fig(density_op)
        model.to(device)

        save_density_fig(density_op)

        print("Generating samples")
        model.eval()
        samples = []
        for i in tqdm(range(100)):
            if args.density_estimator == "flow":
                batch_samples = model.sample(100)
                samples.append(batch_samples)
            else:
                batch_samples, _ = model.sample(100, 5)
                samples.append(batch_samples[0])
        
        all_samples = torch.cat(samples, dim=0)
        ax_samples.clear()
        ax_samples.set_axis_off()
        ax_samples.set_xlim(-4, 4)
        ax_samples.set_ylim(-4, 4)
        ax_samples.scatter(all_samples[:, 0].detach().cpu().numpy(), all_samples[:, 1].detach().cpu().numpy())

        sample_file = os.path.join(data_t_subdir, "samples_{:s}_{:s}.png".format(data_t, args.experiment_id))
        fig_samples.savefig(sample_file, bbox_inches='tight', pad_inches=0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Paths and names
    parser.add_argument("--root", default="toy", type=str)
    parser.add_argument("--experiment_id", default="vanilla", type=str)
    parser.add_argument("--density_estimator", default='kpf', choices=['kpf', 'gmm', 'glow'])
    parser.add_argument("--sample_dir", default="./samples", type=str)

    parser.add_argument("--pf_sample_size", default=10000, type=int)
    args = parser.parse_args()

    args.sample_dir = os.path.join(args.sample_dir, args.root, args.experiment_id)
    
    for _dir in [args.sample_dir]:
        os.makedirs(_dir, exist_ok=True)

    main(args)