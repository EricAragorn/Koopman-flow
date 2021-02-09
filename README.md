# Official repository for Density Approximation in Deep Generative Models with Kernel Transfer Operators
Required packages: PyTorch (1.5+), numpy, jax, neural-tangent, scikit-learn, scikit-image, tqdm, matplotlib, visdom (for visualization)

Due to the use of mixed PyTorch and jax, the code is currently only supported on machines with at least 2 gpus if neural tangent kernel is used (GPU 0 for closed-form kernel computation in jax, and GPU 1 for other PyTorch operations). 

To learn and generate samples from the toy densities using the kernel PF operator, use
```
python flow_toy.py --experiment_id <name> --density_estimator kpf
```
or
```
python flow_toy_spherical.py --experiment_id <name> --density_estimator kpf
```

Image generation is based on 'ex-post' density estimation using kernel PF operator on the latent space of a (regularized sperical) autoencoder. To train a new autoencoder and learn the operator, use
```
python train_pf_generation.py --config configs/<dataset>/<dataset>_sn_spehricial_<density estimator>.yaml
```
