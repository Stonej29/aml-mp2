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

from tqdm import tqdm

from models import GaussianPrior, GaussianEncoder, GaussianDecoder, EnsembleVAE, VAE ,new_encoder, new_decoder
from utils import train, subsample, encode_dataset, create_linspace, log_distances
from geodesics import compute_geodesics_part_b, plot_latent_and_geodesic_piecewise_part_b, plot_latent_and_geodesic_piecewise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, choices=["train", "sample", "eval", "geodesic", "cov", "cov2"],
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
    parser.add_argument("--epochs-per-geodesic", type=int, default=500, help="Optimization epochs for geodesics")
    parser.add_argument("--num-t", type=int, default=100, help="Number of points along the geodesic curve")
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

        test_tensors = datasets.MNIST("data/", train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
        test_data = subsample(test_tensors.data, test_tensors.targets, args.num_train_data, args.num_classes)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


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


        for i in range(linspace_points.shape[0]):
            print(linspace_points[i].shape)
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
            title= f"Geodesic for Ensemble VAE",
        )

        # Then create the plot with decoder uncertainty
        if args.num_decoders > 1:  # Only run if you have multiple decoders
            from geodesics import plot_latent_with_decoder_uncertainty
            
            plot_latent_with_decoder_uncertainty(
                model,
                latent_codes, 
                labels, 
                optimized_points, 
                linspace_points, 
                num_classes,
                args.num_pairs,
                grid_size=300,  # Lower this value if it's too slow
                title=f"Geodesic For Ensemble VAE",
            )

        # Saving distances for CoV
        vae_id = args.model_name.split('_')[-1].split('.')[0]
        log_distances(distances, euc_distances, args.num_decoders, vae_id)
    elif args.mode == "cov":
        num_retrainings = 10  
        num_pairs_cov = 10 
        geodesic_epochs = 500     
        
        mc_num_samples = 200      
        decoder_counts = [1, 2, 3]  #
        fixed_model_decoders = 3  

        all_results = {}  
        
        os.makedirs("figures/cov", exist_ok=True)
        
        test_tensors = datasets.MNIST("data/", train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
        test_data = subsample(test_tensors.data, test_tensors.targets, args.num_train_data, args.num_classes)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        np.random.seed(25)
        total_samples = len(test_data)
        selected_indices = np.random.choice(total_samples, num_pairs_cov * 2, replace=False)
        pair_indices = [(selected_indices[i], selected_indices[i+1]) for i in range(0, len(selected_indices), 2)]
        print("Selected test pair indices:", pair_indices)
        
        for num_decoders in decoder_counts:
            all_results[num_decoders] = {
                "euclidean_dists_all": {pair: [] for pair in pair_indices},
                "geodesic_dists_all": {pair: [] for pair in pair_indices}
            }

        for seed_idx in tqdm(range(1, num_retrainings+1), desc="Processing models"):
            # Only load models with 3 decoders
            model_filename = f"ensemble_model_{fixed_model_decoders}_decoders_{seed_idx}.pt"
            model_path = os.path.join(args.experiment_folder, model_filename)
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found; skipping retraining {seed_idx}")
                continue

            print(f"\nProcessing model {seed_idx + 1}/{num_retrainings}")
            
            # Load the full 3-decoder model
            prior = GaussianPrior(args.latent_dim)
            full_decoders = nn.ModuleList([GaussianDecoder(new_decoder(args.latent_dim)) for _ in range(fixed_model_decoders)])
            encoder = GaussianEncoder(new_encoder(args.latent_dim))
            full_model = EnsembleVAE(prior, full_decoders, encoder).to(device)
            full_model.load_state_dict(torch.load(model_path, map_location=device))
            full_model.eval()
            
            for (idx1, idx2) in pair_indices:
                print(f"  Processing pair ({idx1}, {idx2})")
                
                img1 = test_data[idx1][0].unsqueeze(0).float()
                img2 = test_data[idx2][0].unsqueeze(0).float()
                img1 = img1.to(device)
                img2 = img2.to(device)

                with torch.no_grad():  
                    q1 = full_model.encoder(img1)
                    q2 = full_model.encoder(img2)
                    latent1 = q1.mean.squeeze(0).detach().cpu().numpy()
                    latent2 = q2.mean.squeeze(0).detach().cpu().numpy()

                # Euclidean distance (same for all decoder counts since encoding is identical)
                euc_dist = np.linalg.norm(latent1 - latent2)
                
                # Create initial curve (same for all decoder counts)
                t_np = np.linspace(0, 1, args.num_t)
                curve_np = np.zeros((args.latent_dim, args.num_t))
                for i in range(args.latent_dim):
                    curve_np[i] = latent1[i] * (1 - t_np) + latent2[i] * t_np
                
                for num_decoders in decoder_counts:
                    subset_decoders = nn.ModuleList(list(full_model.decoders)[:num_decoders])
                    
                    curve = torch.tensor(curve_np, dtype=torch.float32, device=device, requires_grad=True)

                    optimized_curve, geo_dist = compute_geodesics_part_b(
                        curve,
                        subset_decoders,  
                        ensemble_size=num_decoders,
                        num_samples=mc_num_samples,
                        epochs=geodesic_epochs,
                        learning_rate=1e-2,
                        seed=None
                    )
                    
                    all_results[num_decoders]["euclidean_dists_all"][(idx1, idx2)].append(euc_dist)
                    all_results[num_decoders]["geodesic_dists_all"][(idx1, idx2)].append(geo_dist.item())
                    
                    print(f"    Using {num_decoders} decoders: Euc dist: {euc_dist:.4f}, Geo dist: {geo_dist.item():.4f}")
                        
        for num_decoders in decoder_counts:
            print(f"\n==== Results for {num_decoders} decoders ====")
            cov_euc = []
            cov_geo = []
            
            for pair in pair_indices:
                dists_euc = np.array(all_results[num_decoders]["euclidean_dists_all"][pair])
                dists_geo = np.array(all_results[num_decoders]["geodesic_dists_all"][pair])
                
                if len(dists_euc) > 0 and np.mean(dists_euc) != 0:
                    cov_euc_val = np.std(dists_euc) / np.mean(dists_euc)
                else:
                    cov_euc_val = 0
                    
                if len(dists_geo) > 0 and np.mean(dists_geo) != 0:
                    cov_geo_val = np.std(dists_geo) / np.mean(dists_geo)
                else:
                    cov_geo_val = 0
                    
                cov_euc.append(cov_euc_val)
                cov_geo.append(cov_geo_val)
                
                print(f"Pair {pair}: Euclidean CoV = {cov_euc_val:.4f}, Geodesic CoV = {cov_geo_val:.4f}")
            
            # Calculate average CoV across all pairs
            avg_cov_euc = np.mean(cov_euc) if len(cov_euc) > 0 else 0
            avg_cov_geo = np.mean(cov_geo) if len(cov_geo) > 0 else 0
            print(f"Average Euclidean CoV: {avg_cov_euc:.4f}, Average Geodesic CoV: {avg_cov_geo:.4f}")
            
            all_results[num_decoders]["cov_euc"] = cov_euc
            all_results[num_decoders]["cov_geo"] = cov_geo
            all_results[num_decoders]["avg_cov_euc"] = avg_cov_euc
            all_results[num_decoders]["avg_cov_geo"] = avg_cov_geo
            
        plt.figure(figsize=(10, 6))
        x = np.arange(len(decoder_counts))
        width = 0.35
        
        # Extract average CoVs for each decoder count
        avg_covs_euc = [all_results[n]["avg_cov_euc"] for n in decoder_counts]
        avg_covs_geo = [all_results[n]["avg_cov_geo"] for n in decoder_counts]
        
        # Plot bars
        plt.bar(x - width/2, avg_covs_euc, width, label="Avg Euclidean CoV", color='skyblue')
        plt.bar(x + width/2, avg_covs_geo, width, label="Avg Geodesic CoV", color='lightgreen')
        # Add labels and formatting
        plt.xlabel("Number of Decoders Used from 3-Decoder Model")
        plt.ylabel("Average Coefficient of Variation (CoV)")
        plt.title("Impact of Ensemble Size on Latent Space Consistency\n(Controlled Experiment)")
        plt.xticks(x, [f"{n} Decoders" for n in decoder_counts])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(avg_covs_euc):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
        for i, v in enumerate(avg_covs_geo):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(f"figures/cov/controlled_cov_comparison.png", dpi=300)
        plt.show()
        
        # Add boxplot comparison
        plt.figure(figsize=(12, 6))
        
        # Create data structures for boxplot
        euc_boxplot_data = [all_results[n]["cov_euc"] for n in decoder_counts]
        geo_boxplot_data = [all_results[n]["cov_geo"] for n in decoder_counts]
        
        # Create subplot for Euclidean CoV
        plt.subplot(1, 2, 1)
        box_euc = plt.boxplot(euc_boxplot_data, patch_artist=True)
        for box in box_euc['boxes']:
            box.set(facecolor='lightblue')
        for median in box_euc['medians']:
            median.set(color='navy', linewidth=1.5)
            
        plt.title('Euclidean CoV Distribution')
        plt.ylabel('Coefficient of Variation')
        plt.xlabel('Number of Decoders Used')
        plt.xticks(range(1, len(decoder_counts) + 1), [f"{n}" for n in decoder_counts])
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        # Create subplot for Geodesic CoV
        plt.subplot(1, 2, 2)
        box_geo = plt.boxplot(geo_boxplot_data, patch_artist=True)
        for box in box_geo['boxes']:
            box.set(facecolor='lightgreen')
        for median in box_geo['medians']:
            median.set(color='darkgreen', linewidth=1.5)
            
        plt.title('Geodesic CoV Distribution')
        plt.ylabel('Coefficient of Variation')
        plt.xlabel('Number of Decoders Used')
        plt.xticks(range(1, len(decoder_counts) + 1), [f"{n}" for n in decoder_counts])
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"figures/cov/controlled_cov_boxplot.png", dpi=300)
        plt.show()
        
        # Print final summary
        print("\n===== FINAL SUMMARY (CONTROLLED EXPERIMENT) =====")
        print("Average CoV by decoder count:")
        for n in decoder_counts:
            print(f"{n} decoders - Euclidean: {all_results[n]['avg_cov_euc']:.4f}, Geodesic: {all_results[n]['avg_cov_geo']:.4f}")