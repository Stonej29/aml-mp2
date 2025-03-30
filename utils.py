import torch
from tqdm import tqdm
import pandas as pd
import os

def train(model, optimizer, data_loader, epochs, device):
    """Train a VAE model."""
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
                optimizer.zero_grad()
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


def subsample(data, targets, num_data, num_classes):
    """Subsample a dataset to only include specific classes and limited samples."""
    idx = targets < num_classes
    new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
    new_targets = targets[idx][:num_data]
    return torch.utils.data.TensorDataset(new_data, new_targets)


def encode_dataset(vae_model, data_loader, device):
    """Encode a dataset through the VAE encoder and return latent codes and labels."""
    all_latents = []
    all_labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            q = vae_model.encoder(x)
            # Use the mean of the latent distribution for visualization
            all_latents.append(q.mean.cpu())
            all_labels.append(y)
    latent_codes = torch.cat(all_latents, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return latent_codes, labels

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

def log_distances(distances, num_decoders, vae_id, output_path="distances_log.csv"):
    """
    Save each distance as a row with num_decoders and vae info.
    """
    if hasattr(distances, 'tolist'):
        distances = distances.tolist()

    df = pd.DataFrame({
        "num_decoders": [num_decoders] * len(distances),
        "vae_id": [vae_id] * len(distances),
        "distance": distances
    })

    # Append to file or create new
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, index=False)