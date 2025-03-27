import torch
from tqdm import tqdm

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