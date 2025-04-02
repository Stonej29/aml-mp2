import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
import os

import os
import matplotlib.patheffects as pe  



def plot_latent_and_geodesic_piecewise_old(latent_codes, labels, optimized_points, linspace_points, num_classes, num_pairs):
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
        plt.savefig(f'test.png', bbox_inches='tight', dpi=300)
        plt.show()

# def compute_geodesics(linspace_points_path,  model, epochs, patience=10, epsilon=1e-5):
#     learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad = True)
#     optimizer = torch.optim.LBFGS([learnable_points], lr=0.01, max_iter=20, history_size=10)                         
#     decoder_net = model.decoder.decoder_net
    
#     best_energy = float('inf')
#     patience_counter = 0

#     def closure():
#         optimizer.zero_grad()
#         x = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1).T  # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input

#         mean_distribution = decoder_net(x) 
#         mean_diff = mean_distribution[:-1] - mean_distribution[1:]                                                           # shape: (num_t-1, 1, 28, 28) calculate the difference 
#         norm = torch.norm(mean_diff, dim=(-1, -2))**2                                                                       # shape: (num_t-1, 1) calculate the squared norm for each latent variable
#         energy = torch.sum(norm, dim=0)                                                                                 # shape: (1,) the sum across the num_t-1 

#         energy.backward()  

#         return energy

#     # Optimize using LBFGS with early stopping
#     print('start')
#     for i in range(epochs):
#             optimizer.step(closure)
#             current_energy = closure().item()

#             if i % 10 == 0:
#                 print(f"Epoch: {i}, Energy: {current_energy}")

#             # Check for improvement
#             if current_energy < best_energy - epsilon:
#                 best_energy = current_energy
#                 patience_counter = 0
#             else:
#                 patience_counter += 1

#             # Early stopping condition
#             if patience_counter >= patience:
#                 print(f"Early stopping at epoch {i}")
#                 break

#     optimized_points = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)  # shape: (num_pairs, M, num_t) save the optimized points for plotting pater
#     diffs = optimized_points[:, 1:] - optimized_points[:, :-1]                                                                          # Compute differences between points
#     distances = torch.norm(diffs, dim=0).sum()                                                                                          # Compute and sum Euclidean distances
#     energy = torch.tensor(current_energy)

#     return optimized_points, distances, energy


def calculate_energy(x, decoder_net):
        mean_distribution = decoder_net(x)
        mean_diff = mean_distribution[:-1] - mean_distribution[1:]
        norm = torch.norm(mean_diff, dim=(-1, -2))**2
        energy = torch.sum(norm, dim=0)
        return energy

def compute_geodesics_Adam(linspace_points_path, model, epochs, lr):
    learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad=True)
    optimizer = torch.optim.Adam([learnable_points], lr=lr)
    decoder_net = model.decoder.decoder_net

    print('start')
    for i in range(epochs):
        optimizer.zero_grad()
        x = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1).T
        
        energy = calculate_energy(x, decoder_net)
        energy.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Epoch: {i}, Energy: {energy.item()}")

    optimized_points = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)
    diffs = optimized_points[:, 1:] - optimized_points[:, :-1]
    distances = torch.norm(diffs, dim=0).sum()
    energy = torch.tensor(energy.item())

    return optimized_points, distances, energy



def compute_geodesics_part_b(linspace_points_path, decoders, ensemble_size, 
                          num_samples=200, epochs=100, learning_rate=0.01, seed=None):
    
    learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad=True)                                          # shape: (num_pairs, num_t-2)
    
    optimizer = torch.optim.Adam([learnable_points], lr=learning_rate)                                                                  # I used adam optimizer instead of LBFGS as it was converging better
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        x = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)            # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input

        
        # MONTE CARLO APPROXIMATION 
        energy = monte_carlo_energy_vectorized(x, decoders, ensemble_size, num_samples, seed)                                              # calculate the energy of the curve using the ensemble energy function
        energy.backward()
        
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch: {i}, Energy: {energy.item():.6f}")
    
    optimized_points = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1) # shape : (num_pairs, M, num_t) save the optimized points for plotting 
    diffs = optimized_points[:, 1:] - optimized_points[:, :-1]                                                                          # Compute differences between points                                            
    distances = torch.norm(diffs, dim=0).sum()                                                                                          # Compute and sum Euclidean distances
    
    return optimized_points, distances

def monte_carlo_ensamble_part_b(curve, decoders, ensemble_size, num_samples=200, seed=None):
    N = curve.shape[-1] - 1                                       
    
    total_energy = 0.0
    
    # pregenerate indices for l and k
        # Save and set random seed if provided
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(seed)

    
    # For each segment of the curve (the summation from i=0 to N)
    for i in range(N):
        l_indices = torch.randint(0, ensemble_size, (num_samples,))
        k_indices = torch.randint(0, ensemble_size, (num_samples,))
        #print(f"First segment {i}")
        # Get current and next points in latent space
        
        #print(f"k_ind{k_indices}")
        #print(f"l_ind{l_indices}")
        t_i = curve[:, i].unsqueeze(0)                              # shape: (batch_dim, D)
        t_i_next = curve[:, i+1].unsqueeze(0)                       # shape: (batch_dim, D)
        
        segment_energy = 0.0
        
        
        for sample in range(num_samples):
            #print(f"Samples {sample}")
            # select index from the pre-generated indices
            l = l_indices[sample].item()
            k = k_indices[sample].item()
            
            # Get the decoder for the selected indices
            f_l = decoders[l]
            f_k = decoders[k]
            
            # Decode the points
            x_l = f_l(t_i).mean  
            x_k = f_k(t_i_next).mean
            
            # Compute the squared difference
            squared_diff = torch.sum((x_l - x_k) ** 2)
            
            # accumulate the energy for this sample
            segment_energy += squared_diff
        
        # average the energy over the number of samples
        segment_energy /= num_samples
        
        # accumulate the energy for this segment
        total_energy += segment_energy
    
    return total_energy

def monte_carlo_energy_vectorized(curve, decoders, ensemble_size, num_samples=200, seed=None):
    """
    Ultimate vectorized implementation of Monte Carlo energy computation
    with fixed random seeding for stable optimization.

    Parameters
    ----------
    curve : torch.Tensor
        Points along the curve in latent space, shape (2, num_points)
    decoders : list
        List of decoder networks
    ensemble_size : int
        Number of decoders to use
    num_samples : int
        Number of Monte Carlo samples
    seed : int, optional
        Random seed for stable sampling

    Returns
    -------
    energy : torch.Tensor
        The estimated energy of the curve
    """
    # Save and set random seed if provided
    if seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
            torch.cuda.manual_seed(seed)

    # Limit to available decoders
    ensemble_size = min(ensemble_size, len(decoders))
    selected_decoders = decoders[:ensemble_size]

    # Get number of segments
    N = curve.shape[1] - 1

    # Generate all random indices at once
    l_indices = torch.randint(0, ensemble_size, (N, num_samples))
    k_indices = torch.randint(0, ensemble_size, (N, num_samples))
    
        # Restore RNG state to avoid affecting global state
    if seed is not None:
        torch.set_rng_state(rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)

    # Prepare batched points
    all_points = curve.transpose(0, 1)  # Shape: (num_points, latent_dim)

    # Run a single point to get output shape
    sample_out = selected_decoders[0](all_points[0:1]).mean
    output_shape = sample_out.shape[1:]  # Exclude batch dimension
    flattened_size = torch.prod(torch.tensor(output_shape)).item()

    # Initialize tensor for all decoder outputs
    all_outputs = torch.zeros(ensemble_size, all_points.shape[0], flattened_size, device=curve.device)

    # Compute all decoder outputs for all points
    batch_size = 64  # Adjust as needed for memory
    for d_idx, decoder in enumerate(selected_decoders):
        for i in range(0, all_points.shape[0], batch_size):
            end_idx = min(i + batch_size, all_points.shape[0])
            batch_points = all_points[i:end_idx]
            outputs = decoder(batch_points).mean
            all_outputs[d_idx, i:end_idx] = outputs.reshape(end_idx - i, -1)

    # Compute energy values in vectorized form
    segment_energies = torch.zeros(N, device=curve.device)
    for i in range(N):
        l_outputs = all_outputs[l_indices[i], i].reshape(num_samples, -1)
        k_outputs = all_outputs[k_indices[i], i+1].reshape(num_samples, -1)
        squared_diffs = torch.sum((l_outputs - k_outputs) ** 2, dim=1)
        segment_energies[i] = torch.mean(squared_diffs)

    total_energy = torch.sum(segment_energies)
    return total_energy



def plot_latent_and_geodesic_piecewise_part_b(latent_codes, labels, optimized_points, linspace_points, num_classes, num_pairs):
    plt.figure(figsize=(8, 8))
    colors = [
        "black", "red", "blue", "green", "darkorange", "purple", "cyan", "magenta", 
        "brown", "yellow", "darkgreen", "lime", "teal", "navy", "pink", "gold", 
        "orangered", "violet", "darkslategray", "mediumspringgreen", "cornflowerblue", 
        "hotpink"

    ]
    colors = plt.cm.tab20(np.linspace(0, 1, num_pairs))

    with torch.no_grad():
        op_np = optimized_points.detach().cpu().numpy()
        lp_np = linspace_points.detach().cpu().numpy()
        
    plt.figure()

    plt.scatter(latent_codes[:, 0], latent_codes[:, 1], s=1, c=labels, alpha=0.5)
    for i in range(op_np.shape[0]):
        color = colors[i]
        plt.plot((op_np[i, 0, 0], op_np[i, 0, -1]), (op_np[i, -1, 0], op_np[i, -1, -1]), 'o', color=color)  # Plot the start and end points
        plt.plot(op_np[i, 0, :], op_np[i, -1, :], '-', color=color)
        plt.plot(lp_np[i, 0, :], lp_np[i, -1, :], '--', color=color)

    plt.title("Ensemble Geodesics in Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/png/geod/ensemble_geodesics_piecewise.png", dpi=300)
    plt.savefig("figures/svg/geod/ensemble_geodesics_piecewise.svg", dpi=300)
    plt.close()


def plot_latent_and_geodesic_piecewise(latent_codes, labels, optimized_points, linspace_points, num_classes, num_pairs, title="Geodesics in Latent Space"):
    plt.figure(figsize=(10, 10))
    
    # Define distinct class colors (can be adjusted)
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green for classes
    
    geodesic_colors = plt.cm.Set1(np.linspace(0, 1, num_pairs))
    
    # Option 2: Use a custom list of distinct colors
   # geodesic_colors = [
   #     "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
   #     "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5"
   # ]
    geodesic_colors = np.array(geodesic_colors)
    
    op_np = optimized_points.detach().cpu().numpy() if torch.is_tensor(optimized_points) else optimized_points.numpy()
    lp_np = linspace_points.detach().cpu().numpy() if torch.is_tensor(linspace_points) else linspace_points.numpy()
    
    for i in range(num_classes):
        class_mask = labels == i
        plt.scatter(latent_codes[class_mask, 0], latent_codes[class_mask, 1], 
                   s=5, c=[class_colors[i]], alpha=0.3, label=f'Class {i}')
    
    # Plot geodesics and euclidean paths
    for i in range(op_np.shape[0]):
        # Cycle through colors using modulus indexing
        color = geodesic_colors[i % len(geodesic_colors)]
        
        start_point = op_np[i, :, 0]
        end_point = op_np[i, :, -1]
        
        start_class = 0  
        end_class = 0
        
        # Find the closest latent code for the start point
        # bit hacky but its to colour the end and start points with the class of the closest point in latent space
        for j in range(len(latent_codes)):
            if np.allclose(latent_codes[j], start_point, rtol=1e-3, atol=1e-3):
                start_class = labels[j]
                break
                
        # Find the closest latent code for the end point
        for j in range(len(latent_codes)):
            if np.allclose(latent_codes[j], end_point, rtol=1e-3, atol=1e-3):
                end_class = labels[j]
                break
        
        # Plot start and end points with class colors
        plt.scatter([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    s=100, 
                    c=[class_colors[int(start_class) % len(class_colors)], 
                       class_colors[int(end_class) % len(class_colors)]],
                    edgecolors='black', 
                    linewidths=1.5, 
                    zorder=5)
        
        # Plot geodesic path with a solid line and add a black outline for contrast
        plt.plot(op_np[i, 0, :], op_np[i, 1, :], '-', 
                 color=color, linewidth=2, zorder=4,
                 path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])
        
        # Plot straight (Euclidean) path with a dashed line (optional: add similar effects)
        plt.plot(lp_np[i, 0, :], lp_np[i, 1, :], '--', 
                 color=color, linewidth=1.5, alpha=0.7, zorder=3)
    
    # Add dummy plots for the legend
    plt.plot([], [], '-', color='black', linewidth=2, label='Geodesic Path')
    plt.plot([], [], '--', color='black', linewidth=1.5, label='Euclidean Path')
    plt.plot([], [], 'o', color='black', markersize=8, markerfacecolor='lightgray', markeredgecolor='black', label='Start/End Points')
    
    plt.title(title, fontsize=34)  #16
    plt.xlabel('z1', fontsize=28)   #14
    plt.ylabel('z2', fontsize=28)   #14
    plt.xticks(fontsize=22)  # Increase this value to make the x-axis tick labels larger
    plt.yticks(fontsize=22)  # Increase this value to make the y-axis tick labels larger

    
    plt.legend(loc='lower left', fontsize=16)
    
    plt.tight_layout()
    
    os.makedirs("figures/png/geod", exist_ok=True)
    os.makedirs("figures/svg/geod", exist_ok=True)
    plt.savefig(f"figures/png/geod/{title}.png", dpi=300)
    plt.savefig(f"figures/svg/geod/{title}.pdf")
    plt.show()