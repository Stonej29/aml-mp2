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
    parser.add_argument("mode", type=str, choices=["train", "sample", "eval", "geodesic", "cov"],
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

        # For peter
        # access list of trained decoders
        #print(model.decoders)
        #print(len(model.decoders))
    

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
            title= f"Geodesic for Ensemble VAE - {str(args.num_decoders)} decoders",
        )


        # Saving distances for CoV
        vae_id = args.model_name.split('_')[-1].split('.')[0]
        log_distances(distances, euc_distances, args.num_decoders, vae_id)
    elif args.mode == "cov":
        num_retrainings = 10      
        num_pairs_cov = 20        
        geodesic_epochs = 300    
        mc_num_samples = 300      
        decoder_counts = [1, 2, 3] 

        all_results = {}  
        
        os.makedirs("figures/cov", exist_ok=True)
        
        test_tensors = datasets.MNIST("data/", train=False, download=True,
                                    transform=transforms.Compose([transforms.ToTensor()]))
        test_data = subsample(test_tensors.data, test_tensors.targets, args.num_train_data, args.num_classes)

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        np.random.seed(42)
        total_samples = len(test_data)
        selected_indices = np.random.choice(total_samples, num_pairs_cov * 2, replace=False)
        pair_indices = [(selected_indices[i], selected_indices[i+1]) for i in range(0, len(selected_indices), 2)]
        print("Selected test pair indices:", pair_indices)
        
        # Loop through different decoder counts
        for num_decoders in decoder_counts:
            print(f"\n==== Processing models with {num_decoders} decoders ====")
            
            euclidean_dists_all = {pair: [] for pair in pair_indices}
            geodesic_dists_all = {pair: [] for pair in pair_indices}

            # Loop over model retrainings
            for seed_idx in range(num_retrainings):
                model_filename = f"ensemble_model_{num_decoders}_decoders_{seed_idx}.pt"
                model_path = os.path.join(args.experiment_folder, model_filename)
                if not os.path.exists(model_path):
                    print(f"Model file {model_path} not found; skipping retraining {seed_idx}")
                    continue

                print(f"Processing model {seed_idx + 1}/{num_retrainings}")
                
                prior = GaussianPrior(args.latent_dim)
                decoders = nn.ModuleList([GaussianDecoder(new_decoder(args.latent_dim)) for _ in range(num_decoders)])
                encoder = GaussianEncoder(new_encoder(args.latent_dim))
                model = EnsembleVAE(prior, decoders, encoder).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                # For each fixed pair, compute distances
                for (idx1, idx2) in pair_indices:
                    print(f"  Processing pair ({idx1}, {idx2})")
                    
                    # Get the images (normalized as during training)
                    img1 = test_data[idx1][0].unsqueeze(0).float() #/ 255.0
                    img2 = test_data[idx2][0].unsqueeze(0).float() #/ 255.0
                    img1 = img1.to(device)
                    img2 = img2.to(device)

                    #  latent codes 
                    with torch.no_grad():  
                        q1 = model.encoder(img1)
                        q2 = model.encoder(img2)
                        latent1 = q1.mean.squeeze(0).detach().cpu().numpy()  # Convert to numpy immediately
                        latent2 = q2.mean.squeeze(0).detach().cpu().numpy()  # Convert to numpy immediately

                    euc_dist = np.linalg.norm(latent1 - latent2)
                    euclidean_dists_all[(idx1, idx2)].append(euc_dist)
                    
                    t_np = np.linspace(0, 1, args.num_t)
                    curve_np = np.zeros((args.latent_dim, args.num_t))
                    
                    for i in range(args.latent_dim):
                        curve_np[i] = latent1[i] * (1 - t_np) + latent2[i] * t_np
                    
                    curve = torch.tensor(curve_np, dtype=torch.float32, device=device, requires_grad=True)

                    # Run geodesic optimization
                    optimized_curve, geo_dist = compute_geodesics_part_b(
                        curve,
                        model.decoders,
                        ensemble_size=num_decoders,  # Use the current decoder count
                        num_samples=mc_num_samples,
                        epochs=geodesic_epochs,
                        learning_rate=1e-2,
                        seed=None  # No fixed seed to ensure fresh samples each time
                    )
                    
                    geodesic_dists_all[(idx1, idx2)].append(geo_dist.item())
                    print(f"    Euclidean distance: {euc_dist:.4f}, Geodesic distance: {geo_dist.item():.4f}")
                        

            # Compute the Coefficient of Variation (CoV) for each pair
            cov_euc = []
            cov_geo = []
            for pair in pair_indices:
                dists_euc = np.array(euclidean_dists_all[pair])
                dists_geo = np.array(geodesic_dists_all[pair])
                
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
                print(f"Pair {pair}: Euclidean distances = {dists_euc}, CoV = {cov_euc_val:.4f}")
                print(f"Pair {pair}: Geodesic distances = {dists_geo}, CoV = {cov_geo_val:.4f}")

            # Calculate and display averages
            avg_cov_euc = np.mean(cov_euc) if len(cov_euc) > 0 else 0
            avg_cov_geo = np.mean(cov_geo) if len(cov_geo) > 0 else 0
            print(f"Average Euclidean CoV with {num_decoders} decoders: {avg_cov_euc:.4f}")
            print(f"Average Geodesic CoV with {num_decoders} decoders: {avg_cov_geo:.4f}")

            # Plot CoV for each pair for this decoder count
            plt.figure(figsize=(8, 6))
            x = np.arange(len(pair_indices))
            width = 0.35
            plt.bar(x - width/2, cov_euc, width, label="Euclidean CoV")
            plt.bar(x + width/2, cov_geo, width, label="Geodesic CoV")
            plt.xlabel("Pair Index")
            plt.ylabel("Coefficient of Variation (CoV)")
            plt.title(f"CoV with {num_decoders} Decoders")
            plt.xticks(x, [f"Pair {i+1}" for i in range(len(pair_indices))])
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"figures/cov/cov_plot_{num_decoders}_decoders.png", dpi=300)
            
            # Store results for this decoder count
            all_results[num_decoders] = {
                "cov_euc": cov_euc,
                "cov_geo": cov_geo,
                "avg_cov_euc": avg_cov_euc,
                "avg_cov_geo": avg_cov_geo,
                "raw_euc_dists": euclidean_dists_all,
                "raw_geo_dists": geodesic_dists_all
            }
        
        # Final comparison plot across decoder counts
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
        plt.xlabel("Number of Decoders in Ensemble VAE")
        plt.ylabel("Average Coefficient of Variation (CoV)")
        plt.title("Impact of Ensemble Size on Latent Space Consistency")
        plt.xticks(x, [f"{n} Decoders" for n in decoder_counts])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(avg_covs_euc):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
        for i, v in enumerate(avg_covs_geo):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(f"figures/cov/cov_comparison_across_decoders.png", dpi=300)
        plt.show()
        
        # Plot individual pair CoVs across decoder counts
        # Calculate grid dimensions based on number of pairs
        n_pairs = len(pair_indices)
        n_cols = 4  # Use 4 columns
        n_rows = (n_pairs + n_cols - 1) // n_cols  # Ceiling division to get required rows
        
        plt.figure(figsize=(16, 4*n_rows))  # Increase figure size to fit all pairs
        
        for i, pair_idx in enumerate(range(n_pairs)):
            plt.subplot(n_rows, n_cols, i+1)
            
            pair_covs_euc = [all_results[n]["cov_euc"][pair_idx] for n in decoder_counts]
            pair_covs_geo = [all_results[n]["cov_geo"][pair_idx] for n in decoder_counts]
            
            plt.plot(decoder_counts, pair_covs_euc, 'o-', label="Euclidean", color='blue')
            plt.plot(decoder_counts, pair_covs_geo, 's-', label="Geodesic", color='green')
            plt.xlabel("Num Decoders")
            plt.ylabel("CoV")
            plt.title(f"Pair {pair_idx+1}")
            if i == 0:
                plt.legend()
                
        plt.tight_layout()
        plt.savefig(f"figures/cov/cov_by_pair_across_decoders.png", dpi=300)
        plt.show()
        
        # Add boxplot comparison of CoV distributions
        plt.figure(figsize=(12, 6))

        # Create data structures for boxplot
        euc_boxplot_data = [all_results[n]["cov_euc"] for n in decoder_counts]
        geo_boxplot_data = [all_results[n]["cov_geo"] for n in decoder_counts]

        # Create subplot for Euclidean CoV
        plt.subplot(1, 2, 1)
        box_euc = plt.boxplot(euc_boxplot_data, patch_artist=True)
        # Customize boxplot colors
        for box in box_euc['boxes']:
            box.set(facecolor='lightblue')
        for median in box_euc['medians']:
            median.set(color='navy', linewidth=1.5)

        plt.title('Euclidean CoV Distribution')
        plt.ylabel('Coefficient of Variation')
        plt.xlabel('Number of Decoders')
        plt.xticks(range(1, len(decoder_counts) + 1), [f"{n}" for n in decoder_counts])
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Create subplot for Geodesic CoV
        plt.subplot(1, 2, 2)
        box_geo = plt.boxplot(geo_boxplot_data, patch_artist=True)
        # Customize boxplot colors
        for box in box_geo['boxes']:
            box.set(facecolor='lightgreen')
        for median in box_geo['medians']:
            median.set(color='darkgreen', linewidth=1.5)

        plt.title('Geodesic CoV Distribution')
        plt.ylabel('Coefficient of Variation')
        plt.xlabel('Number of Decoders')
        plt.xticks(range(1, len(decoder_counts) + 1), [f"{n}" for n in decoder_counts])
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"figures/cov/cov_boxplot_comparison.png", dpi=300)
        plt.show()
        
        # Plot detailed comparison of CoV values for each pair across decoder counts
        # First for Geodesic distances (which should show the ensemble effect more clearly)
        plt.figure(figsize=(14, 6))
        x = np.arange(len(pair_indices))
        width = 0.25  # narrower bars for 3 groups
        
        # Extract CoV values for each decoder count and pair
        geo_covs_by_decoder = {n: all_results[n]["cov_geo"] for n in decoder_counts}
        
        # Plot bars for each decoder count
        for i, n_decoders in enumerate(decoder_counts):
            plt.bar(x + (i-1)*width, geo_covs_by_decoder[n_decoders], width, 
                label=f"{n_decoders} Decoder(s)", alpha=0.7)
        
        # Add labels and formatting
        plt.xlabel("Point Pair Index")
        plt.ylabel("Geodesic CoV")
        plt.title("Detailed Comparison of Geodesic CoV by Pair and Decoder Count")
        plt.xticks(x, [f"Pair {i+1}" for i in range(len(pair_indices))])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add value labels on top of bars
        for i, n_decoders in enumerate(decoder_counts):
            for j, v in enumerate(geo_covs_by_decoder[n_decoders]):
                plt.text(j + (i-1)*width, v + 0.01, f"{v:.2f}", ha='center', fontsize=8, rotation=90)
                
        plt.tight_layout()
        plt.savefig(f"figures/cov/geodesic_cov_by_pair_comparison.png", dpi=300)
        plt.show()
        
        # Calculate and visualize the CoV differences as decoder count increases
        plt.figure(figsize=(14, 6))
        
        # Calculate relative CoV reduction from 1→3 decoders
        geo_cov_reduction = []
        for i in range(len(pair_indices)):
            if geo_covs_by_decoder[1][i] > 0:  # Avoid division by zero
                reduction = (geo_covs_by_decoder[1][i] - geo_covs_by_decoder[3][i]) / geo_covs_by_decoder[1][i] * 100
            else:
                reduction = 0
            geo_cov_reduction.append(reduction)
        
        # Sort pairs by reduction amount (highest first)
        sorted_indices = np.argsort(geo_cov_reduction)[::-1]
        sorted_reductions = [geo_cov_reduction[i] for i in sorted_indices]
        sorted_pair_names = [f"Pair {i+1}" for i in sorted_indices]
        
        # Plot sorted reductions
        bars = plt.bar(range(len(sorted_reductions)), sorted_reductions, color='skyblue')
        
        # Highlight positive reductions (where 3 decoders improved over 1)
        for i, v in enumerate(sorted_reductions):
            if v > 0:
                bars[i].set_color('green')
            else:
                bars[i].set_color('red')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel("Point Pairs (sorted by improvement)")
        plt.ylabel("CoV Reduction from 1→3 Decoders (%)")
        plt.title("Geodesic CoV Reduction by Using 3 Decoders vs 1 Decoder")
        plt.xticks(range(len(sorted_reductions)), sorted_pair_names, rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(sorted_reductions):
            plt.text(i, v + (1 if v >= 0 else -5), f"{v:.1f}%", ha='center')
        
        plt.tight_layout()
        plt.savefig(f"figures/cov/geodesic_cov_improvement_ranking.png", dpi=300)
        plt.show()
        
        # Print final summary
        print("\n===== FINAL SUMMARY =====")
        print("Average CoV by decoder count:")
        for n in decoder_counts:
            print(f"{n} decoders - Euclidean: {all_results[n]['avg_cov_euc']:.4f}, Geodesic: {all_results[n]['avg_cov_geo']:.4f}")
            
        # Print detailed analysis of the pairs with biggest changes
        print("\n==== Pairs with largest CoV reduction (1→3 decoders) ====")
        for i in range(min(3, len(sorted_indices))):
            pair_idx = sorted_indices[i]
            original_pair = pair_indices[pair_idx]
            print(f"Pair {pair_idx+1} (indices {original_pair}):")
            print(f"  1 decoder: CoV = {geo_covs_by_decoder[1][pair_idx]:.4f}")
            print(f"  2 decoders: CoV = {geo_covs_by_decoder[2][pair_idx]:.4f}")
            print(f"  3 decoders: CoV = {geo_covs_by_decoder[3][pair_idx]:.4f}")
            print(f"  Reduction: {sorted_reductions[i]:.2f}%")
    elif args.mode == "cov3":
        # Additional parameters for the CoV experiment
        num_retrainings = 10      # Number of retrainings/models per configuration
        num_pairs_cov = 2         # Number of fixed point pairs for evaluation
        geodesic_epochs = 350     # Number of epochs for geodesic optimization
        mc_num_samples = 200      # Number of Monte Carlo samples for energy estimation
        decoder_counts = [1, 2, 3]  # Run for all these decoder counts

        # Results storage by decoder count
        all_results = {}  # Will store results indexed by decoder count
        
        # Load the test dataset (subset of MNIST with the desired classes)
        test_dataset = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
        indices = test_dataset.targets < args.num_classes
        test_dataset.data = test_dataset.data[indices]
        test_dataset.targets = test_dataset.targets[indices]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Select fixed indices for pairs (using a fixed seed for reproducibility)
        np.random.seed(42)
        total_samples = len(test_dataset.data)
        selected_indices = np.random.choice(total_samples, num_pairs_cov * 2, replace=False)
        pair_indices = [(selected_indices[i], selected_indices[i+1]) for i in range(0, len(selected_indices), 2)]
        print("Selected test pair indices:", pair_indices)
        
        # Loop through different decoder counts
        for num_decoders in decoder_counts:
            print(f"\n==== Processing models with {num_decoders} decoders ====")
            
            # Dictionaries to store distances across retrainings for each pair
            euclidean_dists_all = {pair: [] for pair in pair_indices}
            geodesic_dists_all = {pair: [] for pair in pair_indices}

            # Loop over model retrainings
            for seed_idx in range(num_retrainings):
                model_filename = f"ensemble_model_{num_decoders}_decoders_{seed_idx}.pt"
                model_path = os.path.join(args.experiment_folder, model_filename)
                if not os.path.exists(model_path):
                    print(f"Model file {model_path} not found; skipping retraining {seed_idx}")
                    continue

                print(f"Processing model {seed_idx + 1}/{num_retrainings}")
                
                # Reinitialize and load model architecture from scratch for each iteration
                prior = GaussianPrior(args.latent_dim)
                decoders = nn.ModuleList([GaussianDecoder(new_decoder(args.latent_dim)) for _ in range(num_decoders)])
                encoder = GaussianEncoder(new_encoder(args.latent_dim))
                model = EnsembleVAE(prior, decoders, encoder).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()

                # For each fixed pair, compute distances
                for (idx1, idx2) in pair_indices:
                    print(f"  Processing pair ({idx1}, {idx2})")
                    
                    # Get the images (normalized as during training)
                    img1 = test_dataset.data[idx1].unsqueeze(0).unsqueeze(0).float() #/ 255.0
                    img2 = test_dataset.data[idx2].unsqueeze(0).unsqueeze(0).float() #/ 255.0
                    img1 = img1.to(device)
                    img2 = img2.to(device)

                    # Obtain latent codes (using the mean of the encoder's output)
                    with torch.no_grad():  # No need for gradients when encoding
                        q1 = model.encoder(img1)
                        q2 = model.encoder(img2)
                        latent1 = q1.mean.squeeze(0).detach().cpu().numpy()  # Convert to numpy immediately
                        latent2 = q2.mean.squeeze(0).detach().cpu().numpy()  # Convert to numpy immediately

                    # Compute Euclidean distance (in numpy)
                    euc_dist = np.linalg.norm(latent1 - latent2)
                    euclidean_dists_all[(idx1, idx2)].append(euc_dist)
                    
                    # Create curve in numpy first
                    t_np = np.linspace(0, 1, args.num_t)
                    curve_np = np.zeros((args.latent_dim, args.num_t))
                    
                    for i in range(args.latent_dim):
                        curve_np[i] = latent1[i] * (1 - t_np) + latent2[i] * t_np
                    
                    # Convert to tensor with fresh gradients
                    curve = torch.tensor(curve_np, dtype=torch.float32, device=device, requires_grad=True)

                    try:
                        # Run geodesic optimization
                        optimized_curve, geo_dist = compute_geodesics_part_b(
                            curve,
                            model.decoders,
                            ensemble_size=num_decoders,  # Use the current decoder count
                            num_samples=mc_num_samples,
                            epochs=geodesic_epochs,
                            learning_rate=1e-2,
                            seed=None  # No fixed seed to ensure fresh samples each time
                        )
                        
                        geodesic_dists_all[(idx1, idx2)].append(geo_dist.item())
                        print(f"    Euclidean distance: {euc_dist:.4f}, Geodesic distance: {geo_dist.item():.4f}")
                        
                    except Exception as e:
                        print(f"    ❌ Error computing geodesic: {str(e)}")
                        # Try with shorter optimization as fallback
                        try:
                            print(f"    Attempting fallback with shorter optimization...")
                            # Create fresh curve again
                            curve = torch.tensor(curve_np, dtype=torch.float32, device=device, requires_grad=True)
                            
                            optimized_curve, geo_dist = compute_geodesics_part_b(
                                curve,
                                model.decoders,
                                ensemble_size=num_decoders,  # Use the current decoder count
                                num_samples=50,  # Reduced samples
                                epochs=30,       # Reduced epochs
                                learning_rate=2e-2,  # Higher learning rate
                                seed=None
                            )
                            
                            geodesic_dists_all[(idx1, idx2)].append(geo_dist.item())
                            print(f"    Fallback succeeded! Geodesic distance: {geo_dist.item():.4f}")
                        except Exception as e2:
                            print(f"    ❌ Fallback also failed: {str(e2)}")
                            # If all fails, use Euclidean as approximation
                            geodesic_dists_all[(idx1, idx2)].append(euc_dist * 1.2)  # Crude approximation
                            print(f"    Using approximation instead")
                    
                    # Force cleanup to avoid memory issues
                    torch.cuda.empty_cache() if device == "cuda" else None

            # Compute the Coefficient of Variation (CoV) for each pair
            cov_euc = []
            cov_geo = []
            for pair in pair_indices:
                dists_euc = np.array(euclidean_dists_all[pair])
                dists_geo = np.array(geodesic_dists_all[pair])
                
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
                print(f"Pair {pair}: Euclidean distances = {dists_euc}, CoV = {cov_euc_val:.4f}")
                print(f"Pair {pair}: Geodesic distances = {dists_geo}, CoV = {cov_geo_val:.4f}")

            # Calculate and display averages
            avg_cov_euc = np.mean(cov_euc) if len(cov_euc) > 0 else 0
            avg_cov_geo = np.mean(cov_geo) if len(cov_geo) > 0 else 0
            print(f"Average Euclidean CoV with {num_decoders} decoders: {avg_cov_euc:.4f}")
            print(f"Average Geodesic CoV with {num_decoders} decoders: {avg_cov_geo:.4f}")

            # Plot CoV for each pair for this decoder count
            plt.figure(figsize=(8, 6))
            x = np.arange(len(pair_indices))
            width = 0.35
            plt.bar(x - width/2, cov_euc, width, label="Euclidean CoV")
            plt.bar(x + width/2, cov_geo, width, label="Geodesic CoV")
            plt.xlabel("Pair Index")
            plt.ylabel("Coefficient of Variation (CoV)")
            plt.title(f"CoV with {num_decoders} Decoders")
            plt.xticks(x, [f"Pair {i+1}" for i in range(len(pair_indices))])
            plt.legend()
            plt.tight_layout()
            os.makedirs("figures/cov", exist_ok=True)
            plt.savefig(f"figures/cov/cov_plot_{num_decoders}_decoders.png", dpi=300)
            
            # Store results for this decoder count
            all_results[num_decoders] = {
                "cov_euc": cov_euc,
                "cov_geo": cov_geo,
                "avg_cov_euc": avg_cov_euc,
                "avg_cov_geo": avg_cov_geo
            }
        
        # Final comparison plot across decoder counts
        plt.figure(figsize=(10, 6))
        x = np.arange(len(decoder_counts))
        width = 0.35
        
        # Extract average CoVs for each decoder count
        avg_covs_euc = [all_results[n]["avg_cov_euc"] for n in decoder_counts]
        avg_covs_geo = [all_results[n]["avg_cov_geo"] for n in decoder_counts]
        
        # Plot bars
        plt.bar(x - width/2, avg_covs_euc, width, label="Avg Euclidean CoV")
        plt.bar(x + width/2, avg_covs_geo, width, label="Avg Geodesic CoV")
        
        # Add labels and formatting
        plt.xlabel("Number of Decoders in Ensemble VAE")
        plt.ylabel("Average Coefficient of Variation (CoV)")
        plt.title("Impact of Ensemble Size on Latent Space Consistency")
        plt.xticks(x, [f"{n} Decoders" for n in decoder_counts])
        plt.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(avg_covs_euc):
            plt.text(i - width/2, v + 0.01, f"{v:.3f}", ha='center')
        for i, v in enumerate(avg_covs_geo):
            plt.text(i + width/2, v + 0.01, f"{v:.3f}", ha='center')
            
        plt.tight_layout()
        plt.savefig(f"figures/cov/cov_comparison_across_decoders.png", dpi=300)
        plt.show()
        
        # Plot individual pair CoVs across decoder counts
        # Calculate grid dimensions based on number of pairs
        n_pairs = len(pair_indices)
        n_cols = 4  # Use 4 columns
        n_rows = (n_pairs + n_cols - 1) // n_cols  # Ceiling division to get required rows
        
        plt.figure(figsize=(16, 4*n_rows))  # Increase figure size to fit all pairs
        
        for i, pair_idx in enumerate(range(n_pairs)):
            plt.subplot(n_rows, n_cols, i+1)
            
            pair_covs_euc = [all_results[n]["cov_euc"][pair_idx] for n in decoder_counts]
            pair_covs_geo = [all_results[n]["cov_geo"][pair_idx] for n in decoder_counts]
            
            plt.plot(decoder_counts, pair_covs_euc, 'o-', label="Euclidean", color='blue')
            plt.plot(decoder_counts, pair_covs_geo, 's-', label="Geodesic", color='green')
            plt.xlabel("Num Decoders")
            plt.ylabel("CoV")
            plt.title(f"Pair {pair_idx+1}")
            if i == 0:
                plt.legend()
                
        plt.tight_layout()
        plt.savefig(f"figures/cov/cov_by_pair_across_decoders.png", dpi=300)
        plt.show()

        
        # Print final summary
        print("\n===== FINAL SUMMARY =====")
        print("Average CoV by decoder count:")
        for n in decoder_counts:
            print(f"{n} decoders - Euclidean: {all_results[n]['avg_cov_euc']:.4f}, Geodesic: {all_results[n]['avg_cov_geo']:.4f}")