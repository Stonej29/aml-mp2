import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

def plot_latent_and_geodesic_piecewise(latent_codes, labels, optimized_points, linspace_points, num_classes, num_pairs, args):
    plt.figure(figsize=(8, 8))
    colors = [
        "black", "red", "blue", "green", "darkorange", "purple", "cyan", "magenta", 
        "brown", "yellow", "darkgreen", "lime", "teal", "navy", "pink", "gold", 
        "orangered", "violet", "darkslategray", "mediumspringgreen", "cornflowerblue", 
        "hotpink"

    ]
    colors = plt.cm.tab20(np.linspace(0, 1, num_pairs))

    with torch.no_grad():
        op_np = optimized_points.numpy()  
        lp_np = linspace_points.numpy()

        plt.figure()

        plt.scatter(latent_codes[:, 0], latent_codes[:, 1], s=1, c=labels, alpha=0.5)
        for i in range(op_np.shape[0]):
            color = colors[i]
            plt.plot((op_np[i, 0, 0], op_np[i, 0, -1]), (op_np[i, -1, 0], op_np[i, -1, -1]), 'o', color=color)  # Plot the start and end points
            plt.plot(op_np[i, 0, :], op_np[i, -1, :], '-', color=color)
            plt.plot(lp_np[i, 0, :], lp_np[i, -1, :], '--', color=color)
                
        plt.legend(title="Digits", loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=min(5, num_classes), fontsize=10)
        if args.noise == 'true':
            plt.savefig(f'optimize_folder/results_patience{args.patience}_seed{args.seed}_num_t{args.num_t}_opt{args.optimizer}.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(f'optimize_folder/results_patience{args.patience}_seed{args.seed}_num_t{args.num_t}_opt{args.optimizer}_without_noise.png', bbox_inches='tight', dpi=300)
        plt.show()

def compute_geodesics_LBFGS(linspace_points_path,  model, epochs, patience=10, epsilon=1e-5):
    learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad = True)
    optimizer = torch.optim.LBFGS([learnable_points], lr=0.01, max_iter=20, history_size=10)                         
    decoder_net = model.decoder.decoder_net
    
    best_energy = float('inf')
    patience_counter = 0

    def closure():
        optimizer.zero_grad()
        x = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1).T  # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input

        mean_distribution = decoder_net(x) 
        mean_diff = mean_distribution[:-1] - mean_distribution[1:]                                                           # shape: (num_t-1, 1, 28, 28) calculate the difference 
        norm = torch.norm(mean_diff, dim=(-1, -2))**2                                                                       # shape: (num_t-1, 1) calculate the squared norm for each latent variable
        energy = torch.sum(norm, dim=0)                                                                                 # shape: (1,) the sum across the num_t-1 

        energy.backward()  

        return energy

    # Optimize using LBFGS with early stopping
    print('start')
    for i in range(epochs):
            optimizer.step(closure)
            current_energy = closure().item()

            if i % 10 == 0:
                print(f"Epoch: {i}, Energy: {current_energy}")

            # Check for improvement
            if current_energy < best_energy - epsilon:
                best_energy = current_energy
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping condition
            if patience_counter >= patience:
                print(f"Early stopping at epoch {i}")
                break

    optimized_points = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)  # shape: (num_pairs, M, num_t) save the optimized points for plotting pater
    diffs = optimized_points[:, 1:] - optimized_points[:, :-1]                                                                          # Compute differences between points
    distances = torch.norm(diffs, dim=0).sum()                                                                                          # Compute and sum Euclidean distances
    energy = torch.tensor(current_energy)

    return optimized_points, distances, energy


def compute_geodesics_Adam(linspace_points_path, model, epochs, patience=10, epsilon=1e-5, lr=0.01):
    learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad=True)
    optimizer = torch.optim.Adam([learnable_points], lr=lr)
    decoder_net = model.decoder.decoder_net

    best_energy = float('inf')
    patience_counter = 0
    
    print('start')
    for i in range(epochs):
        optimizer.zero_grad()
        x = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1).T

        mean_distribution = decoder_net(x)
        mean_diff = mean_distribution[:-1] - mean_distribution[1:]
        norm = torch.norm(mean_diff, dim=(-1, -2))**2
        energy = torch.sum(norm, dim=0)

        energy.backward()
        optimizer.step()

        current_energy = energy.item()

        if i % 10 == 0:
            print(f"Epoch: {i}, Energy: {current_energy}")

        # Check for improvement
        if current_energy < best_energy - epsilon:
            best_energy = current_energy
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping condition
        if patience_counter >= patience:
            print(f"Early stopping at epoch {i}")
            break

    optimized_points = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)
    diffs = optimized_points[:, 1:] - optimized_points[:, :-1]
    distances = torch.norm(diffs, dim=0).sum()
    energy = torch.tensor(current_energy)

    return optimized_points, distances, energy

# Example usage
# Assuming `model` and `linspace_points_path` are defined
# optimized_points, distances, energy = compute_geodesics_Adam(linspace_points_path, model, epochs=100)
