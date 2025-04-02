import os
import torch
import argparse
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

from models import GaussianPrior, GaussianEncoder, GaussianDecoder, VAE, new_encoder, new_decoder
from geodesics_old import plot_latent_and_geodesic_piecewise, plot_latent_and_geodesic_polynomial
from geodesics import compute_geodesics_LBFGS, compute_geodesics_Adam, plot_latent_and_geodesic_piecewise
from utils import train, subsample, encode_dataset, create_linspace

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesic", "geodesic_poly"],
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
        "--epochs",
        type=int,
        default=500,
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
        "--num-segments",
        type=int,
        default=10,
        metavar="N",
        help="number of segments for the geodesic curve (default: %(default)s)",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=4,
        metavar="N",
        help="degree of polynomial for geodesic (default: %(default)s)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=50,
        metavar="N",
        help="number of points for polynomial geodesic (default: %(default)s)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="seed to create reproducible results (default: %(default)s)",
    )

    parser.add_argument(
        "--num-pairs",
        type=int,
        default=25,
        metavar="N",
        help="num pairs to plot and find geodesics for default: %(default)s)",
    )

    parser.add_argument(
        "--epochs-per-geodesic",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs for geodesic (default: %(default)s)",
    )

    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    parser.add_argument(
        "--epsilon",  # number of points along the curve
        type=float,
        default=1e-5,
        help="difference between previous energy and current energy path optimization (default: %(default)s)",
    )

    parser.add_argument(
        "--patience",  # number of points along the curve
        type=int,
        default=10,
        metavar="N",
        help="number of times the energy is not optimized before break stopping the optimization loop (default: %(default)s)",
    )

    parser.add_argument(
        "--optimizer",  # number of points along the curve
        type=str,
        default='adam',
        metavar="N",
        help="the used optimizer (default: %(default)s)",
    )

    
    parser.add_argument(
        "--noise",  # number of points along the curve
        type=str,
        default='true',
        metavar="N",
        help="if the noise is constant in the training (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load a subset of MNIST and create data loaders
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

    # Define model
    M = args.latent_dim

    if args.mode == "train":
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs,
            device,
            args.noise
        )
        if args.noise == 'true':
            torch.save(
                model.state_dict(),
                f"{experiments_folder}/model_seed{args.seed}.pt"
            )
        else:
            torch.save(
                model.state_dict(),
                f"{experiments_folder}/model_seed{args.seed}_without_noise.pt"
            )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt", map_location=device))
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
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt", map_location=device))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesic_old":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt", map_location=device))
        model.eval()
        
        test_dataset = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
        
        indices = test_dataset.targets < num_classes
        test_dataset.data = test_dataset.data[indices]
        test_dataset.targets = test_dataset.targets[indices]
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        latent_codes, labels = encode_dataset(model, test_loader, device)
        
        plot_latent_and_geodesic_piecewise(model, latent_codes, labels, N_segments=args.num_segments)
    
    elif args.mode == "geodesic_poly":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + f"/model_seed{args.seed}.pt", map_location=device))
        model.eval()
        
        test_dataset = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
        
        indices = test_dataset.targets < num_classes
        test_dataset.data = test_dataset.data[indices]
        test_dataset.targets = test_dataset.targets[indices]
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        latent_codes, labels = encode_dataset(model, test_loader, device)
        
        #plot_latent_and_geodesic_polynomial(model, latent_codes, labels, degree=args.poly_degree, N_points=args.num_points)
        plot_latent_and_geodesic_piecewise

    elif args.mode == "geodesic": # most up to date
        np.random.seed(42)
        torch.manual_seed(42)

        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder(M)),
            GaussianEncoder(new_encoder(M)),
        ).to(device)

        if args.noise == 'true':
            model.load_state_dict(torch.load(args.experiment_folder + f"/model_seed{args.seed}.pt"))
        else:
            print('without_noise')
            model.load_state_dict(torch.load(args.experiment_folder + f"/model_seed{args.seed}_without_noise.pt"))

        model.eval()

        # Data loading 
        test_dataset = datasets.MNIST("data/", train=False, download=True, transform=transforms.ToTensor())
        indices = test_dataset.targets < num_classes
        test_dataset.data = test_dataset.data[indices]
        test_dataset.targets = test_dataset.targets[indices]
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Encode our original dataset into the latent space
        latent_codes, labels = encode_dataset(model, test_loader, device)

        random_indices = np.random.choice(len(latent_codes), args.num_pairs * 2, replace=False)

        selected_latent_codes = latent_codes[random_indices] # Shape: (50, 2)
        points = np.hstack([selected_latent_codes[:args.num_pairs, :], selected_latent_codes[args.num_pairs:, :]])  # Shape: (args.num_pairs, 4)
        points = torch.tensor(points, dtype=torch.float32)  

        linspace_points = create_linspace(points, args.num_t)                                                           # get equal spaced points from the start to end position with args.num_t points

        # Define empty tensors for book keeping
        optimized_points = torch.zeros_like(linspace_points)                                                            # placeholder for optimized results
        distances = torch.zeros(linspace_points.shape[0])                                                               # placeholder for the calculated distances
        energies = torch.zeros(linspace_points.shape[0])  
        
        if args.optimizer == 'LBFGS':
            for i in range(linspace_points.shape[0]):
                optimized_points[i], distances[i], energies[i] = compute_geodesics_LBFGS(linspace_points[i], model, epochs=args.epochs_per_geodesic, patience=args.patience, epsilon=args.epsilon)
        else:
            print('ADAM!')
            for i in range(linspace_points.shape[0]):
                optimized_points[i], distances[i], energies[i] = compute_geodesics_Adam(linspace_points[i], model, epochs=args.epochs_per_geodesic, patience=args.patience, epsilon=args.epsilon)
            
        # Save results to a text file
        if args.noise == 'true':
            with open(f"optimize_folder/results_patience{args.patience}_seed{args.seed}_num_t{args.num_t}_opt{args.optimizer}_.txt", "w") as file:
                file.write("Distances:\n")
                file.write(str(distances) + "\n\n")
                file.write("Energies:\n")
                file.write(str(energies) + "\n")

        else:
            with open(f"optimize_folder/results_patience{args.patience}_seed{args.seed}_num_t{args.num_t}_opt{args.optimizer}_without_noise.txt", "w") as file:
                file.write("Distances:\n")
                file.write(str(distances) + "\n\n")
                file.write("Energies:\n")
                file.write(str(energies) + "\n")

        plot_latent_and_geodesic_piecewise(latent_codes, labels, optimized_points, linspace_points, num_classes, args.num_pairs, args)

