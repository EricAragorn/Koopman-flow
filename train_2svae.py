import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import time
import os
import visdom
import types
import yaml
from skimage.io import imsave
from PIL import Image

plt.ion()

from modules.autoencoder import VAE, TwoStageVAE
from modules.loss import mix_rbf_mmd, poly_mmd, LapLoss
from modules.utils import SubsetDataset
from modules.kernel_fn import *
from dataset import get_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

device = 'cuda'

def get_samples(dataset, sample_size=-1):
    inds = np.random.permutation(len(dataset))
    if sample_size >= 0:
        inds = inds[:sample_size]
    return torch.stack([dataset[ind][0] for ind in inds]).float()

def pre_process(image):
    return image * 2 - 1

def post_process(logits):
    return torch.clamp((logits + 1) / 2, 0., 1.)

def train_two_stage_vae(ae, train_loader, epochs, lr, lr_epochs, lr_frac, args, device=torch.device('cuda'), viz=None):
    parallel_ae = nn.DataParallel(ae).to(device)
    optim = torch.optim.Adam(parallel_ae.parameters(), lr=lr)
    lr_lambda = lambda epoch: np.power(lr_frac, int(epoch) // lr_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    HALF_LOG_TWO_PI = 0.91893
    loggamma_x = ae.loggamma_x
    criterion = lambda x, y: torch.sum(torch.square((x - y) / torch.exp(loggamma_x)) / 2.0 + loggamma_x + HALF_LOG_TWO_PI) / x.size(0)
    for epoch in range(epochs):
        counter = 0
        elapsed_time = 0.
        
        train_iter = tqdm(train_loader)
        train_iter.set_description("Epoch {:d}".format(epoch + 1))
        for i, (X, y) in enumerate(train_iter):
            X = pre_process(X).float().to(device)

            start_time = time.time()
            base_latents, rec_latents, kld_loss = parallel_ae(X)

            rec_loss = criterion(torch.cat(base_latents, dim=-1), torch.cat(rec_latents, dim=-1))
            KLD_loss = kld_loss.mean()
            loss = rec_loss + KLD_loss

            optim.zero_grad()
            loss.backward()
            optim.step()

            elapsed_time += time.time() - start_time
            counter += 1

            train_iter.set_postfix({'Rec Loss': rec_loss.item(), 'KLD Loss': KLD_loss.item()})

            def _visualize(_viz):
                vis_count = np.prod(args.visdim)

                with torch.no_grad():
                    samples, _, _ = ae.sample(vis_count)

                sample_img = post_process(samples).view(-1, *args.img_dim).detach().cpu()

                _viz.images(sample_img[:vis_count], nrow=args.visdim[0], win='samples', opts={'title': 'samples'})

            if i % args.log_iter == 0:
                if viz is not None:
                    _visualize(viz)

        scheduler.step()

        ae_state = ae.state_dict()
        torch.save(ae_state, os.path.join(args.save_dir, "{}-2svae.pth".format(args.experiment_id)))

def main(args):
    device = 'cuda'

    def get_transform(img_dim):
        t = transforms.Compose([
            transforms.Resize((*img_dim[-2:],)),
            transforms.ToTensor(),
        ])
        return t

    if args.visualize:
        viz = visdom.Visdom(port=8097)
    else:
        viz = None

    train_ds = get_dataset(args, get_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.workers)

    # Train Autoencoder
    base_ae = VAE(args.img_dim,
                  encoding_depth=args.ae_encoding_depth, 
                  style=args.dataset,
                  groups=args.ae_groups,
                  use_sn=False)
    two_stage_ae = TwoStageVAE(base_ae, latent_dim=args.ae_encoding_depth)
    if args.two_stage_ae_load_path is not None:
        state_dict = torch.load(args.two_stage_ae_load_path)
        two_stage_ae.load_state_dict(state_dict)
        two_stage_ae.to(device)
    else:
        print("Two stage AE load path is None. Training a new two stage AE...")

        if args.base_ae_load_path is not None:
            print("Loading base AE...")
            state_dict = torch.load(args.base_ae_load_path)
            base_ae.load_state_dict(state_dict)
        else:
            print("Base AE load path is None. Training a new base AE...")
            from train_vae import train_vae
            train_vae(base_ae, train_loader, args.epochs1, args.lr1, args.lr_epochs1, args.lr_frac1, args, device=device, viz=viz)
        train_two_stage_vae(two_stage_ae, train_loader, args.epochs2, args.lr2, args.lr_epochs2, args.lr_frac2, args, device=device, viz=viz)

    two_stage_ae.to(device)
    base_ae.to(device)

    fig_samples = plt.figure()
    fig_samples.set_size_inches(15, 15)
    grid_samples = ImageGrid(fig_samples, 111, args.visdim, axes_pad=0.)
    fig_samples.suptitle("Sampled")

    figs = [fig_samples]
    
    print("Generating samples")
    two_stage_ae.eval()
    for i in tqdm(range(10)):
        samples, _, _ = two_stage_ae.sample(np.prod(args.visdim))
        samples = post_process(samples).view(-1, *args.img_dim).permute(0, 2, 3, 1).detach().cpu().numpy()

        for j, ax in enumerate(np.ravel(grid_samples)):
            ax.clear()
            ax.imshow(np.squeeze(np.squeeze(np.clip(samples[j], 0., 1.))), cmap=plt.cm.gray)
            ax.axis('off')
        
        for f in figs:
            f.canvas.draw()
            f.canvas.flush_events()

        sample_file = os.path.join(args.sample_dir, "sample_{:02d}.png".format(i))
        fig_samples.savefig(sample_file)
    
    img_subdir = os.path.join(args.sample_dir, "images")
    if not os.path.exists(img_subdir):
        os.mkdir(img_subdir)
    counter = 0
    latent_list = []
    for i in tqdm(range(200)):
        with torch.no_grad():
            samples, base_latents, _ = two_stage_ae.sample(50)
            latent_list.append(base_latents)
            samples = post_process(samples).view(-1, *args.img_dim).permute(0, 2, 3, 1).detach().cpu().numpy()

        for s in samples:
            imsave(os.path.join(img_subdir, "sampled_{:d}.png".format(counter)), np.round(s * 255).astype(np.uint8))
            counter += 1

    img_subdir = os.path.join(args.sample_dir, "images_rec")
    if not os.path.exists(img_subdir):
        os.mkdir(img_subdir)
    loader = DataLoader(train_ds, batch_size=50, drop_last=False, pin_memory=True, num_workers=args.workers)
    data_iter = iter(loader)
    counter = 0
    rec_latent_list = []
    for i in tqdm(range(200)):
        X, y = next(data_iter)
        with torch.no_grad():
            X = pre_process(X).float().to(device)
            reconstructions, latents, _ = two_stage_ae.base_vae(X)
            rec_latent_list.append(latents)
            reconstructions = post_process(reconstructions).view(-1, *args.img_dim).permute(0, 2, 3, 1).detach().cpu().numpy()

        for r in reconstructions:
            imsave(os.path.join(img_subdir, "reconstructed_{:d}.png".format(counter)), np.round(r * 255).astype(np.uint8))
            counter += 1
    
    def _concat_latent(latent_batched_list):
        _tmp = [torch.cat(l, dim=0) for l in zip(*latent_batched_list)]
        return torch.cat([_t.view(_t.size(0), -1) for _t in _tmp], dim=-1)

    all_y = _concat_latent(rec_latent_list)
    all_latent = _concat_latent(latent_list)
    mmd_score = mix_rbf_mmd(all_y, all_latent, sigma_list=[1.])
    print("MMD:", mmd_score.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, type=str)
    # Paths and names
    parser.add_argument("--experiment_id", default="vanilla", type=str)
    parser.add_argument("--save_dir", default="./saved_models", type=str)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--sample_dir", default="./samples", type=str)

    # Misc params
    parser.add_argument("--visdim", default=(8, 8), type=int, nargs='+')
    parser.add_argument("--visualize", default=False, action="store_true")

    # Dataset params
    parser.add_argument("--dataset", default="mnist", type=str)
    parser.add_argument("--workers", default=8, type=int)

    # Autoencoder params
    parser.add_argument("--train_ae", default=False, action="store_true")
    parser.add_argument("--base_ae_load_path", default=None, type=str)
    parser.add_argument("--two_stage_ae_load_path", default=None, type=str)
    parser.add_argument("--ae_depths", default=[32, 64, 128, 256, 512], nargs='+', type=int)
    parser.add_argument("--ae_scale_factor", default=1, type=int)
    parser.add_argument("--ae_layers", default=[1, 1, 1, 1], nargs='+', type=int)
    parser.add_argument("--ae_encoding_depth", default=128, type=int)
    parser.add_argument("--ae_groups", default=1, type=int)
    parser.add_argument("--ae_input_noise_var", default=0., type=float)

    # Training hyperparamters
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--epochs1", default=100, type=int)
    parser.add_argument("--lr1", default=1e-4, type=float)
    parser.add_argument("--lr_epochs1", default=40, type=int)
    parser.add_argument("--lr_frac1", default=0.5)
    parser.add_argument("--epochs2", default=100, type=int)
    parser.add_argument("--lr2", default=1e-4, type=float)
    parser.add_argument("--lr_epochs2", default=40, type=int)
    parser.add_argument("--lr_frac2", default=0.5)
    
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            args = types.SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))

    args.sample_dir = os.path.join(args.sample_dir, args.experiment_id)
    
    for _dir in (args.save_dir, args.cache_dir, args.sample_dir):
        os.makedirs(_dir, exist_ok=True)
    
    print(args)
    main(args)