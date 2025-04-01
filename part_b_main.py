import os
import torch
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import GaussianPrior, GaussianEncoder, GaussianDecoder, EnsembleVAE, VAE ,new_encoder, new_decoder
from utils import train, subsample, encode_dataset, create_linspace, log_distances
from geodesics import compute_geodesics_part_b, plot_latent_and_geodesic_piecewise_part_b, plot_latent_and_geodesic_piecewise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "sample", "eval", "geodesic"],
                        help="Mode: train, sample, eval, or geodesic")
    parser.add_argument("--experiment-folder", type=str, default="ensemble_experiment",
                        help="Folder for saving/loading model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Device to use")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--latent-dim", type=int, default=2, help="Dimension of latent variable")
    parser.add_argument("--num-decoders", type=int, default=3, help="Number of decoders in ensemble")
    parser.add_argument("--num-train-data", type=int, default=2048, help="Number of training samples")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of classes to use")
    parser.add_argument("--num-pairs", type=int, default=25, help="Number of latent point pairs for geodesics")
    parser.add_argument("--epochs-per-geodesic", type=int, default=50, help="Optimization epochs for geodesics")
    parser.add_argument("--num-t", type=int, default=20, help="Number of points along the geodesic curve")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--model-name", type=str, default=None, help="Model name")
    args = parser.parse_args()

    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    num_classes = 3
    
    # Load a subset of MNIST data
    train_tensors = datasets.MNIST("data/", train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))
    test_tensors = datasets.MNIST("data/", train=False, download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))
    train_data = subsample(train_tensors.data, train_tensors.targets, args.num_train_data, args.num_classes)
    test_data = subsample(test_tensors.data, test_tensors.targets, args.num_train_data, args.num_classes)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    M = args.latent_dim

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = f"ensemble_model_{args.num_decoders}_decoders.pt"

    prior = GaussianPrior(M)
    decoders = nn.ModuleList([GaussianDecoder(new_decoder(M)) for _ in range(args.num_decoders)])
    encoder = GaussianEncoder(new_encoder(M))
    model = EnsembleVAE(prior, decoders, encoder).to(device)

    if args.mode == "train":
        os.makedirs(args.experiment_folder, exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        train(model, optimizer, train_loader, args.epochs, device)
        torch.save(model.state_dict(), os.path.join(args.experiment_folder, model_name))
    
    elif args.mode == "sample":
        model.load_state_dict(torch.load(os.path.join(args.experiment_folder, model_name), map_location=device))
        model.eval()
        with torch.no_grad():
            samples = model.sample(64).cpu()
            save_image(samples.view(64, 1, 28, 28), "figures/ensemble_samples.png")
            data = next(iter(test_loader))[0].to(device)

            # we use the first one, instead we could average maybe
            recon = model.decoders[0](model.encoder(data).mean).mean
            save_image(torch.cat([data.cpu(), recon.cpu()], dim=0), "figures/ensemble_reconstruction.png")
    
    elif args.mode == "eval":
        model.load_state_dict(torch.load(os.path.join(args.experiment_folder, model_name), map_location=device))
        model.eval()
        elbos = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                elbos.append(model.elbo(x))
        mean_elbo = torch.tensor(elbos).mean()
        print("Mean test ELBO:", mean_elbo.item())
    
    elif args.mode == "geodesic":
        model.load_state_dict(torch.load(os.path.join(args.experiment_folder, model_name), map_location=device))
        model.eval()
        test_dataset = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
        indices = test_dataset.targets < args.num_classes
        test_dataset.data = test_dataset.data[indices]
        test_dataset.targets = test_dataset.targets[indices]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        latent_codes, labels = encode_dataset(model, test_loader, device)
        random_indices = np.random.choice(len(latent_codes), args.num_pairs * 2, replace=False)
        selected_latent_codes = latent_codes[random_indices]

        points = np.hstack([selected_latent_codes[:args.num_pairs, :], selected_latent_codes[args.num_pairs:, :]])
        points = torch.tensor(points, dtype=torch.float32)
        linspace_points = create_linspace(points, args.num_t).to(device)
        optimized_points = torch.zeros_like(linspace_points)
        distances = torch.zeros(linspace_points.shape[0])
        
        z1 = points[:, :args.latent_dim]
        z2 = points[:, args.latent_dim:]

        # Euclidean distances
        euc_distances = torch.norm(z1 - z2, dim=1)

        # For peter
        # access list of trained decoders
        #print(model.decoders)
        #print(len(model.decoders))
    

        for i in range(linspace_points.shape[0]):
            optimized_points[i], distances[i] = compute_geodesics_part_b(linspace_points[i], model.decoders, epochs=args.epochs_per_geodesic, ensemble_size=args.num_decoders, num_samples=100, seed=args.seed)
            print(f"Distances at step {i}: {distances[i]}")
            
        #plot_latent_and_geodesic_piecewise_part_b(latent_codes, labels, optimized_points, linspace_points, num_classes, args.num_pairs)
        plot_latent_and_geodesic_piecewise(
            latent_codes, 
            labels, 
            optimized_points, 
            linspace_points, 
            num_classes,
            args.num_pairs,
            title= f"Geodesic for Ensemble VAE - {str(args.num_decoders)} decoders",
        )


        # Saving distances for CoV
        vae_id = args.model_name.split('_')[-1].split('.')[0]
        log_distances(distances, euc_distances, args.num_decoders, vae_id)

###########################################################################################################
############################ RUN THE CODE WITH THE FOLLOWING COMMANDS #####################################
# 3 decoders
##  python part_b_main.py geodesic   --experiment-folder ensemble_experiment --model-name ensemble_model_3_decoders_1.pt   --device cuda   --num-pairs 10   --epochs-per-geodesic 50   --num-t 20   --num-decoders 3   --latent-dim 2
# 2 decoders
##  python part_b_main.py geodesic   --experiment-folder ensemble_experiment --model-name ensemble_model_2_decoders_1.pt   --device cuda   --num-pairs 10   --epochs-per-geodesic 50   --num-t 20   --num-decoders 2   --latent-dim 2
# 1 decoder
##  python part_b_main.py geodesic   --experiment-folder ensemble_experiment --model-name ensemble_model_1_decoder_1.pt   --device cuda   --num-pairs 10   --epochs-per-geodesic 50   --num-t 20   --num-decoders 1   --latent-dim 2
############################################################################################################