import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn

def plot_latent_and_geodesic_piecewise(latent_codes, labels, optimized_points, linspace_points, num_classes, num_pairs):
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

def compute_geodesics(linspace_points_path,  model, epochs, patience=10, epsilon=1e-5):
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

<<<<<<< HEAD
    return optimized_points, distances



def compute_geodesics_part_b(linspace_points_path, decoders, ensemble_size, 
                          num_samples=200, epochs=100, learning_rate=0.01):
    
    learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad=True)                                          # shape: (num_pairs, num_t-2)
    
    optimizer = torch.optim.Adam([learnable_points], lr=learning_rate)                                                                  # I used adam optimizer instead of LBFGS as it was converging better
    
    for i in range(epochs):
        optimizer.zero_grad()
        
        x = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)            # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input

        
        # MONTE CARLO APPROXIMATION 
        energy = monte_carlo_ensamble_part_b(x, decoders, ensemble_size, num_samples)                                              # calculate the energy of the curve using the ensemble energy function
        energy.backward()
        
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch: {i}, Energy: {energy.item():.6f}")
    
    optimized_points = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1) # shape : (num_pairs, M, num_t) save the optimized points for plotting 
    diffs = optimized_points[:, 1:] - optimized_points[:, :-1]                                                                          # Compute differences between points                                            
    distances = torch.norm(diffs, dim=0).sum()                                                                                          # Compute and sum Euclidean distances
    
    return optimized_points, distances

def monte_carlo_ensamble_part_b(curve, decoders, ensemble_size, num_samples=200):
    N = curve.shape[-1] - 1                                       
    
    total_energy = 0.0
    
    # pregenerate indices for l and k
    l_indices = torch.randint(0, ensemble_size, (num_samples,))
    k_indices = torch.randint(0, ensemble_size, (num_samples,))
    
    # For each segment of the curve (the summation from i=0 to N)
    for i in range(N):
        # Get current and next points in latent space
        
        t_i = curve[:, i].unsqueeze(0)                              # shape: (batch_dim, D)
        t_i_next = curve[:, i+1].unsqueeze(0)                       # shape: (batch_dim, D)
        
        segment_energy = 0.0
        
        
        for sample in range(num_samples):
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
=======
    return optimized_points, distances, energy
>>>>>>> 754eaacd47fb762c4bdfff661d18b11cb0eda35c
