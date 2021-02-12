import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import time
import os
import visdom
import yaml
from skimage.io import imsave
from PIL import Image
import types

plt.ion()

from modules.autoencoder import *
from modules.latent_sampler import *
from modules.loss import mix_rbf_mmd, poly_mmd, LapLoss
from modules.utils import SubsetDataset
from modules.kernel_fn import *
from modules.preimage import *
from modules.flow import *
from dataset import get_dataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

# device = 'cuda'

def get_samples(dataset, sample_size=-1):
    inds = np.random.permutation(len(dataset))
    if sample_size >= 0:
        inds = inds[:sample_size]
    return torch.stack([dataset[ind][0] for ind in inds]).float()

def pre_process(image):
    return image * 2 - 1

def post_process(logits):
    return (logits + 1) / 2

def train_ae(ae, train_loader, epochs, lr, lr_epochs, lr_frac, args, device=torch.device('cuda'), viz=None):
    if args.loss_type == 'l2':
        criterion = lambda x, y: torch.mean(torch.sum((x - y).pow(2), dim=1))
    elif args.loss_type == 'l1':
        criterion = lambda x, y: torch.mean(torch.sum((x - y).abs(), dim=1))
    elif args.loss_type == 'nll':
        def nll_loss(x, y):
            x = post_process(x)
            y = post_process(y)
            return -torch.mean(torch.sum(y * torch.log(torch.clamp_min(x, 1e-12)) + (1 - y) * torch.log(torch.clamp_min(1 - x, 1e-12)), dim=1))

        criterion = nll_loss

    parallel_ae = nn.DataParallel(ae).to(device)
    optim = torch.optim.Adam(parallel_ae.parameters(), lr=lr)

    lr_lambda = lambda epoch: np.power(lr_frac, int(epoch) // lr_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    for epoch in range(epochs):
        train_iter = tqdm(train_loader)
        train_iter.set_description("Epoch {:d}".format(epoch + 1))
        for i, (X, y) in enumerate(train_iter):
            X = pre_process(X).float().to(device)
            y = y.to(device)

            reconstruction, embeddings = parallel_ae(X)
            rec_loss = criterion(reconstruction, X)
            loss = rec_loss
            tqdm_postfix = {'Rec loss': rec_loss.item()}

            # Regularization loss for l2/gradient penalty
            reg_loss = 0.
            if args.reg_type == 'l2':
                if args.reg_decoder_only:
                    param_iter = ae.decoder.parameters()
                else:
                    param_iter  = ae.parameters()
                
                for param in param_iter:
                    reg_loss += param.norm(p='2')
            elif args.reg_type == 'gp':
                raise NotImplementedError

            # Penalty on the l2 norm of embeddings
            emb_pen = 0.
            if not args.ae_spherical_latent:
                for emb in embeddings:
                    emb_pen += emb.square().mean()

            loss += args.reg_lambda * reg_loss + args.embedding_penalty_lambda * emb_pen

            optim.zero_grad()
            loss.backward()
            optim.step()

            train_iter.set_postfix(tqdm_postfix)

            if i % args.log_iter == 0:
                def _visualize(_viz):
                    # Switch temporarily to eval mode
                    ae.eval()

                    with torch.no_grad():
                        interp = ae.bilinear_interpolate(embeddings, steps=args.visdim[0])

                    X_img = post_process(X).view(-1, *args.img_dim).detach().cpu()
                    pred_img = post_process(reconstruction).view(-1, *args.img_dim).detach().cpu()
                    interp_img = post_process(interp).view(args.visdim[0] * args.visdim[0], *args.img_dim).detach().cpu()

                    vis_count = np.prod(args.visdim)

                    _viz.images(X_img[:vis_count], nrow=args.visdim[0], win='x', opts={'title': 'data'})
                    _viz.images(pred_img[:vis_count], nrow=args.visdim[0], win='pred', opts={'title': 'prediction'})
                    _viz.images(interp_img[:vis_count], nrow=args.visdim[0], win='interp', opts={'title': 'Interpolation'})

                    ae.train()

                if viz is not None:
                    _visualize(viz)

        scheduler.step()

        ae_state = ae.state_dict()
        torch.save(ae_state, os.path.join(args.save_dir, "{}-ae.pth".format(args.experiment_id)))

def main(args):
    device = 'cuda'

    global viz
    if args.visualize:
        viz = visdom.Visdom(port=8097)
    else:
        viz = None

    def get_transform(img_dim):
        t = transforms.Compose([
            transforms.Resize((*img_dim[-2:],)),
            transforms.ToTensor(),
        ])
        return t

    train_ds = get_dataset(args, get_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.workers)

    if args.ae_spherical_latent:
        AEModule = SAE
    else:
        AEModule = VanillaAE

    # Train Autoencoder
    ae = AEModule(input_dim=args.img_dim,
                  encoding_depth=args.ae_encoding_depth, 
                  style=args.dataset,
                  use_sn=args.reg_type == "sn",
                  sn_decoder_only=args.reg_decoder_only)
    print(ae)

    if args.ae_load_path is not None:
        ae_state = torch.load(args.ae_load_path)
        ae.load_state_dict(ae_state)
    else:
        print("AE load path is None. Training a new AE...")
        train_ae(ae, train_loader, args.epochs, args.lr, args.lr_epochs, args.lr_frac, args, device=device, viz=viz)

    torch.cuda.empty_cache()
    device = 'cuda:1'
    ae.to(device)
    # ae.eval()

    latent_dim = args.ae_encoding_depth * args.ae_groups

    cached = args.de_use_cached_features
    ignore_labels = False
    sample_size = args.de_sample_size

    pf_train_ds = SubsetDataset(train_ds, sample_size, shuffle=True)

    _cache_file = os.path.join(args.cache_dir, "{:s}-kernelPF-features.pth".format(args.experiment_id))
    if cached and os.path.exists(_cache_file):
        print("Restoring cached features from {:s}...".format(_cache_file))
        features, labels = torch.load(_cache_file)
    else:
        print("Getting features from dataset...")
        loader = DataLoader(pf_train_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=args.workers)
        feature_list = []
        label_list = []
        with torch.no_grad():
            for i, (X, y) in enumerate(tqdm(loader)):
                X = pre_process(X).float().to(device)
                feature_batch = ae.encode(X)
                # feature_batch = ae.get_latents(X)[0]

                feature_list.append(feature_batch)
                if not ignore_labels:
                    if type(y) not in (list, tuple):
                        y = [y]
                    label_list.append(y)
        features = [torch.cat(f, dim=0) for f in zip(*feature_list)]
        labels = [torch.cat(l, dim=0) for l in zip(*label_list)]
        print("Cacheing features for later use...")
        torch.save((features, labels), _cache_file)

    plt.hist(labels[0].cpu().numpy())
    plt.show()

    features_cat = torch.cat(features, dim=-1)
    features_sqr = features_cat.square().sum(-1).unsqueeze(-1)
    features_sqr_dist = features_sqr - 2 * features_cat.mm(features_cat.T) + features_sqr.T
    n = features_cat.size(0)
    sigma = (features_sqr_dist.sum() / (n * (n - 1))).sqrt().item()

    sigma_search_list = [sigma / 2 ** i for i in range(8)]

    all_y = torch.cat([_y.view(-1, args.ae_encoding_depth) for _y in features], dim=-1)[:5000]

    topk = 5

    def eval_model(element, model):
        latent_list = []
        with torch.no_grad():
            for i in range(50):
                latent, latent_inds = model.sample(100, topk)
                latent_list.append(latent)
        _tmp = [torch.cat(l, dim=0) for l in zip(*latent_list)]
        all_latent = torch.cat([_t.view(_t.size(0), -1) for _t in _tmp], dim=-1)
        mmd_score = poly_mmd(all_y, all_latent, deg=3)
        print("{}, MMD:{:f}".format(element, mmd_score.item()))
        return mmd_score

    model = GMMGenerator(latent_samples=features, n_components=10, device=device)
    baseline_score = eval_model('Baseline', model)

    best_sig = -1
    best_score = 10
    for _sig in sigma_search_list:
        input_kernel = NeuralKernel(latent_dim, 10000, n_layers=8, kernel_type='ntk', activation='relu')
        output_kernel = MixRBFKernel(latent_dim, [_sig])
        
        with torch.no_grad():
            model = KernelPFGenerator(input_kernel=input_kernel,
                                        output_kernel=output_kernel,
                                        output_samples=features,
                                        preimage_module=GeodesicInterpPreimage() if args.ae_spherical_latent else WeightedMeanPreimage(),
                                        # preimage_module=MDSPreimage(),
                                        spherical=args.ae_spherical_latent,
                                        labels=None if ignore_labels else labels, 
                                        nystrom_compression=args.pf_use_compression, 
                                        nystrom_points=args.pf_nystrom_points,
                                        epsilon=args.pf_reg_factor, 
                                        p_dim=args.pf_sampling_dim, 
                                        device=device).to(device)
        score = eval_model(_sig, model)
        if score < best_score:
            best_sig = _sig
            best_score = score

    def data_batch_iterator(features):
        datasize = len(features_cat)
        batchsize = 64
        while True:
            index = np.random.permutation(datasize)
            for j in range(datasize // batchsize):
                batch_data = features_cat[index[j * batchsize: (j + 1) * batchsize]]
                yield batch_data

    start = time.time()
    if args.sampler_type == "kpf":
        input_kernel = NeuralKernel(args.pf_sampling_dim, 10000, n_layers=8, kernel_type='ntk', activation='erf')
        output_kernel = MixRBFKernel(latent_dim, [best_sig])

        model = KernelPFGenerator(input_kernel=input_kernel,
                                    output_kernel=output_kernel,
                                    output_samples=features,
                                    preimage_module=GeodesicInterpPreimage() if args.ae_spherical_latent else WeightedMeanPreimage(),
                                    spherical=args.ae_spherical_latent,
                                    labels=None if ignore_labels else labels, 
                                    nystrom_compression=args.pf_use_compression, 
                                    nystrom_points=args.pf_nystrom_points,
                                    epsilon=args.pf_reg_factor, 
                                    p_dim=args.pf_sampling_dim, 
                                    device=device).to(device)
    elif args.sampler_type == "gmm":
        model = GMMGenerator(latent_samples=features, n_components=10, device=device)
    elif args.sampler_type == "glow":
        model = Glow1d(args.ae_encoding_depth, 512, 20).to(device)
        features_cat = torch.cat(features, dim=-1)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, verbose=True)
        
        
        max_steps = 10000
        steps = 0

        _iter = tqdm(enumerate(data_batch_iterator(features_cat)), total=max_steps)
        for i, batch_data in _iter:
            logp = model(batch_data)
            loss = -logp.mean()

            _iter.set_postfix({"Bits per dim": loss.item()})

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            scheduler.step(loss)

            steps += 1
            if steps >= max_steps:
                break
            
            if (i % 100) == 0:
                torch.save((i, model.state_dict()), "./saved_models/{:s}-glow.pth".format(args.experiment_id))
        torch.save((i, model.state_dict()), "./saved_models/{:s}-glow.pth".format(args.experiment_id))
    elif args.sampler_type == "vae":
        model = VAE1d(args.ae_encoding_depth).to(device)

        features_cat = torch.cat(features, dim=-1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=500, verbose=True)

        criterion = lambda x, y: torch.sum(torch.square((x - y))) / x.size(0)

        max_steps = 50000
        steps = 0

        _iter = tqdm(enumerate(data_batch_iterator(features_cat)), total=max_steps)
        for i, batch_data in _iter:
            rec, kld_loss = model(batch_data)

            rec_loss = criterion(rec, batch_data)
            KLD_loss = kld_loss.mean()
            
            loss = rec_loss + 0.01 * KLD_loss

            _iter.set_postfix({"rec loss": rec_loss.item(), "kld loss": KLD_loss.item()})

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            scheduler.step(loss)

            if ((i + 1) % 100) == 0:
                torch.save((i, model.state_dict()), "./saved_models/{:s}-vae.pth".format(args.experiment_id))

            steps += 1
            if steps >= max_steps:
                break
        torch.save((i, model.state_dict()), "./saved_models/{:s}-vae.pth".format(args.experiment_id))

    print("Elapsed time: {:4f}".format(time.time() - start))

    fig_samples = plt.figure()
    fig_samples.set_size_inches(15, 15)
    grid_samples = ImageGrid(fig_samples, 111, args.visdim, axes_pad=0.)
    fig_samples.suptitle("Sampled")

    fig_nearest = plt.figure()
    fig_nearest.set_size_inches(15, 3 * topk)
    grid_nearest = ImageGrid(fig_nearest, 111, (5, topk + 1), axes_pad=0.)
    fig_nearest.tight_layout()

    figs = [fig_samples, fig_nearest]

    def imshow_axis(ax, img):
        ax.clear()
        ax.imshow(img, cmap=plt.cm.gray)
        ax.axis('off')

    print("Generating samples")    
    # Generate visualizations
    model.eval()
    for i in tqdm(range(10)):
        if args.sampler_type == 'glow':
            latent = [model.sample(np.prod(args.visdim))]
        elif args.sampler_type == 'vae':
            latent = [model.sample(np.prod(args.visdim), device=device)]
        else:
            latent, latent_inds = model.sample(np.prod(args.visdim), topk)
        samples = ae.decode(latent)
        samples = post_process(samples).view(-1, *args.img_dim).permute(0, 2, 3, 1).detach().cpu().numpy()

        for j, ax in enumerate(np.ravel(grid_samples)):
            imshow_axis(ax, np.squeeze(np.clip(samples[j], 0., 1.)))
        
        if args.sampler_type not in ['glow', 'vae']:
            for j in range(5):
                imshow_axis(grid_nearest[j * (topk + 1)], np.squeeze(np.clip(samples[j], 0., 1.)))
                for k in range(topk):
                    imshow_axis(grid_nearest[j * (topk + 1) + k + 1], pf_train_ds[latent_inds[j, k]][0].permute(1, 2, 0).squeeze().detach().cpu().numpy())
        
        for f in figs:
            f.canvas.draw()
            f.canvas.flush_events()

        sample_file = os.path.join(args.sample_dir, "sample_{:02d}.png".format(i))
        fig_samples.savefig(sample_file)

        if args.sampler_type not in ['glow', 'vae']:
            nearest_file = os.path.join(args.sample_dir, "nearest_{:02d}.png".format(i))
            fig_nearest.savefig(nearest_file)
    
    # Generate samples for various gamma values (# neighbours used in weighted mean)
    gamma_list = [5, 10, 50, 100]
    for g in gamma_list:
        img_subdir = os.path.join(args.sample_dir, "images_gamma{:d}".format(g))
        if not os.path.exists(img_subdir):
            os.mkdir(img_subdir)
        counter = 0
        latent_list = []

        elapsed_time = 0.
        for i in tqdm(range(100)):
            with torch.no_grad():
                start = time.time()
                if args.sampler_type == 'glow':
                    latent = [model.sample(np.prod(args.visdim))]
                elif args.sampler_type == 'vae':
                    latent = [model.sample(np.prod(args.visdim), device=device)]
                else:
                    latent, _ = model.sample(100, topk=g)
                elapsed_time += time.time() - start
            latent_list.append(latent)
            samples = ae.decode(latent)
            samples = post_process(samples).view(-1, *args.img_dim).permute(0, 2, 3, 1).detach().cpu().numpy()

            for s in samples:
                imsave(os.path.join(img_subdir, "sampled_{:d}.png".format(counter)), np.round(s * 255).astype(np.uint8), check_contrast=False)
                counter += 1
        print("Average sampling time {:f}".format(elapsed_time / (10000)))
    
    # Generate Image reconstructions
    img_subdir = os.path.join(args.sample_dir, "images_rec")
    if not os.path.exists(img_subdir):
        os.mkdir(img_subdir)
    counter = 0
    for i in tqdm(range(200)):
        latent = [_y[i * 50: (i + 1) * 50] for _y in model.y_list]
        samples = ae.decode(latent)
        samples = post_process(samples).view(-1, *args.img_dim).permute(0, 2, 3, 1).detach().cpu().numpy()

        for s in samples:
            imsave(os.path.join(img_subdir, "sampled_{:d}.png".format(counter)), np.round(s * 255).astype(np.uint8))
            counter += 1
    
    # Calculate latent feature mmd using deg 3 polynomial kernel 
    all_y = torch.cat([_y.view(model.y_size, -1) for _y in model.y], dim=-1)
    _tmp = [torch.cat(l, dim=0) for l in zip(*latent_list)]
    all_latent = torch.cat([_t.view(_t.size(0), -1) for _t in _tmp], dim=-1)
    mmd_score = poly_mmd(all_y, all_latent, deg=3)
    print("MMD:", mmd_score.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # if config is not None, override settings with config profile
    parser.add_argument("--config", default=None)
    # Paths and names
    parser.add_argument("--experiment_id", default="vanilla", type=str)
    parser.add_argument("--save_dir", default="./saved_models", type=str)
    parser.add_argument("--save_freq", default=1, type=int)
    parser.add_argument("--cache_dir", default="./cache", type=str)
    parser.add_argument("--sample_dir", default="./samples", type=str)

    # Visualization params
    parser.add_argument("--visdim", default=(8, 8), type=int, nargs='+')
    parser.add_argument("--visualize", default=False, action="store_true")

    # Dataset params
    parser.add_argument("--dataset", default="mnist", choices=['mnist', 'cifar10', 'celeba64'])
    parser.add_argument("--workers", default=8, type=int)

    # Autoencoder params
    parser.add_argument("--ae_load_path", default=None, type=str)
    parser.add_argument("--ae_depths", default=[32, 64, 128, 256], nargs='+', type=int)
    parser.add_argument("--ae_scale_factor", default=1, type=int)
    parser.add_argument("--ae_layers", default=[2, 2, 2], nargs='+', type=int)
    parser.add_argument("--ae_encoding_depth", default=64, type=int)
    parser.add_argument("--ae_groups", default=1, type=int)
    parser.add_argument("--ae_spherical_latent", default=True, type=bool)

    # Flow model params
    parser.add_argument("--pf_use_cached_features", default=False, action='store_true')
    parser.add_argument("--pf_sample_size", default=10000, type=int)
    parser.add_argument("--pf_use_compression", default=False, action='store_true')
    parser.add_argument("--pf_nystrom_points", default=4000, type=int)

    # Training hyperparamters
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=70, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_epochs", default=30, type=int)
    parser.add_argument("--lr_frac", default=0.5)
    parser.add_argument("--loss_type", default="l2", choices=['l1', 'l2', 'nll'])
    parser.add_argument("--reg_type", default=None, choices=['sn', 'l2', 'gp'])
    parser.add_argument("--reg_lambda", default=1e-3, type=float)
    parser.add_argument("--reg_decoder_only", default=True, type=bool)
    parser.add_argument("--embedding_penalty_lambda", default=1e-3, type=float)
    
    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, 'r') as f:
            args = types.SimpleNamespace(**yaml.load(f, Loader=yaml.FullLoader))

    args.sample_dir = os.path.join(args.sample_dir, args.experiment_id)
    
    for _dir in (args.save_dir, args.cache_dir, args.sample_dir):
        os.makedirs(_dir, exist_ok=True)
    print(args)

    main(args)