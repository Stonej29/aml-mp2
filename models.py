import torch
import torch.nn as nn
import torch.distributions as td
import random

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """Define a Gaussian prior distribution with zero mean and unit variance."""
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """Return the prior distribution."""
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """Define a Gaussian encoder distribution based on a given encoder network."""
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """Given a batch of data, return a Gaussian distribution over the latent space."""
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """Define a Gaussian decoder distribution based on a given decoder network."""
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """Given a batch of latent variables, return a Gaussian distribution over the data space."""
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """Define a Variational Autoencoder (VAE) model."""

    def __init__(self, prior, decoder, encoder):
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """Compute the ELBO for the given batch of data."""
        q = self.encoder(x)
        z = q.rsample()
        return torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )

    def sample(self, n_samples=1):
        """Sample from the model."""
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """Compute the negative ELBO for the given batch of data."""
        return -self.elbo(x)

class EnsembleVAE(nn.Module):
    def __init__(self, prior, decoders, encoder):
        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoders)  
        self.encoder = encoder
        self.num_decoders = len(decoders)
    
    def forward(self, x):
        q = self.encoder(x)
        z = q.rsample()
        # Sample one decoder uniformly
        decoder_idx = random.randint(0, self.num_decoders - 1)
        decoder = self.decoders[decoder_idx]
        recon_distribution = decoder(z)

        # elbo loss
        loss = - (recon_distribution.log_prob(x) - q.log_prob(z) + self.prior().log_prob(z))
        return loss.mean()

    def sample(self, n_samples=1):
        """Sample from the model using a randomly selected decoder."""
        z = self.prior().sample(torch.Size([n_samples]))
        decoder_idx = random.randint(0, self.num_decoders - 1)
        decoder = self.decoders[decoder_idx]
        return decoder(z).sample()
    
    def elbo(self, x):
        """Compute the ELBO using a randomly selected decoder."""
        q = self.encoder(x)
        z = q.rsample()
        decoder_idx = random.randint(0, self.num_decoders - 1)
        decoder = self.decoders[decoder_idx]
        return torch.mean(decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z))

# Network architectures
def new_encoder(M):
    """Create a new encoder network for MNIST images."""
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


def new_decoder(M):
    """Create a new decoder network for MNIST images."""
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