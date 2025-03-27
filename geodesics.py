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

def compute_geodesics(linspace_points_path,  model, epochs):
    learnable_points = nn.Parameter(linspace_points_path[:, 1:-1].clone(), requires_grad = True)
    optimizer = torch.optim.LBFGS([learnable_points], lr=0.01, max_iter=20, history_size=10)                         
    decoder_net = model.decoder.decoder_net

    def closure():
        optimizer.zero_grad()
        x = torch.cat(( linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1).T  # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input

        mean_distribution = decoder_net(x) 
        mean_diff = mean_distribution[:-1] - mean_distribution[1:]                                                           # shape: (num_t-1, 1, 28, 28) calculate the difference 
        norms = torch.norm(mean_diff, dim=(-1, -2))**2                                                                       # shape: (num_t-1, 1) calculate the squared norm for each latent variable
        total_norm = torch.sum(norms, dim=0)                                                                                 # shape: (1,) the sum across the num_t-1 

        total_norm.backward()  

        return total_norm

    # Optimize using LBFGS
    for i in range(epochs):
        optimizer.step(closure)  
        if i % 10 == 0:
            print(f"Epoch: {i}, Energy: {closure().item()}")

    optimized_points = torch.cat((linspace_points_path[:, 0][:, None], learnable_points, linspace_points_path[:, -1][:, None]), dim=1)  # shape: (num_pairs, M, num_t) save the optimized points for plotting pater
    diffs = optimized_points[:, 1:] - optimized_points[:, :-1]                                                                          # Compute differences between points
    distances = torch.norm(diffs, dim=0).sum()                                                                                          # Compute and sum Euclidean distances

    return optimized_points, distances