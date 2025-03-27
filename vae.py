# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import einops

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break

def data_subset(aggregate_latents, labels, samples_per_class=500):
    
    # Select balanced samples from each class
    unique_labels = np.unique(labels)
    
    # Create arrays to store balanced MNIST samples
    balanced_latents = []
    balanced_labels = []
    
    for label in unique_labels:
        # Find indices for this class
        indices = np.where(labels == label)[0]
        
        # If there are fewer samples than requested, use all available samples
        if len(indices) <= samples_per_class:
            selected_indices = indices
        else:
            # Otherwise, randomly sample the requested number
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        
        # Append selected samples to our lists
        balanced_latents.append(aggregate_latents[selected_indices])
        balanced_labels.append(np.full(len(selected_indices), label))
    
    # Combine the balanced samples
    balanced_latents = np.vstack(balanced_latents)
    balanced_labels = np.concatenate(balanced_labels)

    return balanced_latents, balanced_labels, unique_labels

def get_log_probs(x_lim, y_lim, prior, resolution=1000):
    # Create grid for contour plot
    x = np.linspace(x_lim[0], x_lim[1], resolution)
    y = np.linspace(y_lim[0], y_lim[1], resolution)
    X, Y = np.meshgrid(x, y)
    grid_points_original = np.stack([X.ravel(), Y.ravel()], axis=-1)

    # Convert to torch tensor for prior evaluation
    grid_tensor = torch.tensor(grid_points_original, dtype=torch.float32)
    
    # Evaluate log probabilities
    with torch.no_grad():
        log_probs = prior.log_prob(grid_tensor).numpy()
    
    # Reshape for plotting
    log_probs = log_probs.reshape(X.shape) # shape: (resolution, resolution)

    return log_probs

def optimize_path(pairs, num_t):
    print(pairs.shape)
    x = torch.linspace(pairs[:, :2], pairs[:,2:], num_t) #https://stackoverflow.com/questions/32208359/is-there-a-multi-dimensional-version-of-arange-linspace-in-numpy

    # print(x)
        # print(l_check.shape)
        # out = model.decoder(l_check)
        # mean_distribution = out.base_dist.loc  # This is the mean of the distribution
        # print("Mean of the distribution:")
        # print(mean_distribution.shape)

        # m = mean_distribution[:-1] - mean_distribution[1:]
        # print('norm')
        # norm = torch.norm(m, dim=(-1, -2))
        # print(torch.sum(norm, dim=0))



def create_linspace(points, num_points):
    # https://stackoverflow.com/questions/49306677/is-there-something-like-np-linspace-for-3d-lines
    """
    Create a linspace between start and end points in 3D using PyTorch,
    given a single tensor where the first half are start points and the second half are end points.

    Parameters:
    - points: 2Nx3 tensor where the first N rows are start points and the last N rows are end points.
    - num_points: Number of points to generate between start and end points.

    Returns:
    - linspace_points: Nx(num_points)x3 tensor of interpolated points.
    """
    N = points.shape[1] // 2
    p1 = points[:, :N]
    p2 = points[:, N:]
    
    # Create a tensor of shape (num_points, 1) with values ranging from 0 to 1
    l1 = torch.linspace(0, 1, num_points)

    # Calculate the interpolated points using broadcasting
    linspace_points = p1[:, :, None] + (p2 - p1)[:, :, None] * l1[None, None, :]
    return linspace_points

def load_points(path):
        loaded_points = []
        with open(path, 'r') as file:
            for line in file:
                row = list(map(float, line.strip().split()))
                loaded_points.append(row)

        # Convert the loaded points into a PyTorch tensor
        return torch.tensor(loaded_points)


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="seed to create reproducible results (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)


    elif args.mode == "geodesics":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        # points = load_points('points.txt')                                                               # Load the points from the text file
        # linspace_points = create_linspace(points, args.num_t)                                            # get equal spaced points from the start to end position with args.num_t points
        # learnable_points = torch.nn.Parameter(linspace_points[:, :, 1:-1].clone(), requires_grad=True)  # convert the numbers in between the start and endpoint to parameter for optimization. Clone so we don't change the original linspace
        # optimized_points = torch.zeros_like(linspace_points)                                             # create zero tensor to store the optimized points
        # optimizer = torch.optim.Adam([learnable_points], lr=0.1)                                         # Initialize the optimizer outside the loop                         

        # # Loop through each of the point pairs
        # for i in range(linspace_points.shape[0]):
        #     for j in range(args.epochs_per_decoder):
        #         optimizer.zero_grad() 
        #         x = torch.cat((linspace_points[i, :, 0][:, None], learnable_points[i], linspace_points[i, :, -1][:, None]), dim=1).T # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input
        #         print(x.shape)
        #         mean_distribution = model.decoder(x).base_dist.loc                                                                   # shape: (num_t, 1, 28, 28) get the mean for the distribution
        #         mean_diff = mean_distribution[:-1] - mean_distribution[1:]                                                           # shape: (num_t-1, 1, 28, 28) calculate the difference 
        #         norms = torch.norm(mean_diff, dim=(-1, -2))**2                                                                       # shape: (num_t-1, 1) calculate the squared norm for each latent variable
        #         total_norm = torch.sum(norms, dim=0)                                                                                 # shape: (1,) the sum across the num_t-1 

        #         total_norm.backward()  
        #         optimizer.step()  
        #     optimized_points[i] = x.T                                                                                                # shape: (num_pairs, M, num_t) save the optimized points for plotting pater


        # Load points and process
        points = load_points('points.txt')                                                                              # load the points from the text file
        linspace_points = create_linspace(points, args.num_t)                                                           # get equal spaced points from the start to end position with args.num_t points
        learnable_points = torch.nn.Parameter(linspace_points[:, :, 1:-1].clone(), requires_grad=True)                  # convert the numbers in between the start and endpoint to parameter for optimization. Clone so we don't change the original linspace
        optimized_points = torch.zeros_like(linspace_points)                                                            # placeholder for optimized results
        distances = torch.zeros(linspace_points.shape[0])                                                               # placeholder for the calculated distances
        optimizer = torch.optim.LBFGS([learnable_points], lr=0.1, max_iter=20, history_size=10)                         # Use LBFGS optimizer

        # Loop through each of the point pairs
        for i in range(linspace_points.shape[0]):
            def closure():
                optimizer.zero_grad()
                x = torch.cat(( linspace_points[i, :, 0][:, None], learnable_points[i], linspace_points[i, :, -1][:, None]), dim=1).T  # shape: (num_t, M) concatenate the first and last point to the learnable points and transpose to match the decoders input
                mean_distribution = model.decoder(x).base_dist.loc                                                                   # shape: (num_t, 1, 28, 28) get the mean for the distribution
                mean_diff = mean_distribution[:-1] - mean_distribution[1:]                                                           # shape: (num_t-1, 1, 28, 28) calculate the difference 
                norms = torch.norm(mean_diff, dim=(-1, -2))**2                                                                       # shape: (num_t-1, 1) calculate the squared norm for each latent variable
                total_norm = torch.sum(norms, dim=0)                                                                                 # shape: (1,) the sum across the num_t-1 

                total_norm.backward()  # Compute gradients
                return total_norm

            # Optimize using LBFGS
            for _ in range(args.epochs_per_decoder):
                optimizer.step(closure)  

            optimized_points[i] = torch.cat((linspace_points[i, :, 0][:, None], learnable_points[i], linspace_points[i, :, -1][:, None]), dim=1)  # shape: (num_pairs, M, num_t) save the optimized points for plotting pater
            diffs = optimized_points[i, :, 1:] - optimized_points[i, :, :-1]                                                                      # Compute differences between points
            distances[i] = torch.norm(diffs, dim=0).sum()                                                                                         # Compute and sum Euclidean distances

        print(distances)

        with torch.no_grad():
            aggregate_latents = []
            aggregate_labels = []
            for batch in mnist_train_loader:
                x = batch[0].to(device)
                labels = batch[1].cpu().numpy()
                q = model.encoder(x)
                z = q.rsample()
                aggregate_latents.append(z.cpu().numpy())
                aggregate_labels.append(labels)
            aggregate_latents = np.concatenate(aggregate_latents, axis=0)
            aggregate_labels = np.concatenate(aggregate_labels, axis=0)
                    
            x_lim = (-6, 6)
            y_lim = (-1, 10)                    
            prior_instance = model.prior.forward() 
            balanced_latents, balanced_labels, unique_labels = data_subset(aggregate_latents, aggregate_labels)
            log_probs = get_log_probs(x_lim, y_lim, prior_instance)

            # Create figure
            plt.figure(figsize=(10, 9))
            plt.xlabel('Latent Dimension 1', fontsize=12)
            plt.ylabel('Latent Dimension 2', fontsize=12)
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            
            # Plot balanced data points with class colors
            cmap = plt.get_cmap('tab10')
            for i, label in enumerate(unique_labels):
                idx = balanced_labels == label
                plt.scatter(balanced_latents[idx, 0],
                            balanced_latents[idx, 1],
                            color=cmap(i), alpha=0.7, s=10,
                            label=f'Class {label}')
                
            colors = [
                "black", "red", "blue", "green", "darkorange", "purple", "cyan", "magenta", 
                "brown", "yellow", "darkgreen", "lime", "teal", "navy", "pink", "gold", 
                "orangered", "violet", "darkslategray", "mediumspringgreen", "cornflowerblue", 
                "hotpink", "darkkhaki", "mediumorchid", "firebrick"
            ]

            op_np = optimized_points.numpy()  # Convert the optimized points to numpy for plotting
            lp_np = linspace_points.numpy()

            for i in range(op_np.shape[0]):
                color = colors[i]
                plt.plot((op_np[i, 0, 0], op_np[i, 0, -1]), (op_np[i, -1, 0], op_np[i, -1, -1]), 'o', color=color)  # Plot the start and end points
                plt.plot(op_np[i, 0, :], op_np[i, -1, :], '-', color=color)
                plt.plot(lp_np[i, 0, :], lp_np[i, -1, :], '--', color=color)
                    
            plt.legend(title="Digits", loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=min(5, num_classes), fontsize=10)
            plt.savefig(f'test.png', bbox_inches='tight', dpi=300)
            plt.show()


            
            
