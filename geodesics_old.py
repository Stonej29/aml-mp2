import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def polynomial_curve(t, c0, c1, weights):
    K = weights.shape[0] + 1
    w_K = -weights.sum(dim=0)  # shape (2,)
    poly_term = torch.zeros_like(c0)
    for i in range(1, K):
        poly_term += weights[i-1] * (t ** i)
    poly_term += w_K * (t ** K)
    return (1 - t) * c0 + t * c1 + poly_term

def energy_piecewise(c0, c1, between_points, decoder_net, N_segments):
    pts_latent = torch.cat([c0.unsqueeze(0), between_points, c1.unsqueeze(0)], dim=0)
    decoded_pts = decoder_net(pts_latent) # means 
    
    #dt = 1.0 / (N_segments)# - 1) # check this with TAs

    E = 0.0
    for i in range(N_segments):
        diff = decoded_pts[i+1] - decoded_pts[i]
        E += (diff**2).sum() #* dt # TODO: ask TAs, do we multiply here?
    return E

def energy_polynomial(c0, c1, weights, decoder_net, N_points):
    t_vals = torch.linspace(0.0, 1.0, N_points)
    dt = 1.0 / (N_points) #- 1) # check this with TAs
    E = 0.0
    for i in range(N_points - 1):
        p1 = polynomial_curve(t_vals[i], c0, c1, weights)
        p2 = polynomial_curve(t_vals[i+1], c0, c1, weights)
        #print(p1.shape)
        #print("unqsuzeezed", p1.unsqueeze(0).shape)
        x1 = decoder_net(p1.unsqueeze(0))
        x2 = decoder_net(p2.unsqueeze(0))

        diff = x2 - x1
        E += (diff ** 2).sum() #* dt # TODO: ask TAs, do we multiply here?
    return E

def compute_geodesic_polynomial(c0, c1, decoder_net, degree=4, N_points=50, epochs=1000, lr=0.001):
    weights = torch.zeros(degree - 1, 2, dtype=torch.float32, requires_grad=True)
    # so weights are w_0, ..., w_{K-1}
    optimizer = optim.Adam([weights], lr=lr)
    
    decoder_net.eval()
    for param in decoder_net.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = energy_polynomial(c0, c1, weights, decoder_net, N_points)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Energy = {loss.item():.4f}")
    
    t_vals = torch.linspace(0.0, 1.0, N_points)
    curve = torch.stack([polynomial_curve(t, c0, c1, weights) for t in t_vals])
    return curve.detach().cpu().numpy(), weights.detach()

def compute_geodesic_piecewise(c0, c1, decoder_net, N_segments=20, epochs=100, lr=0.001):
    xs = torch.linspace(c0[0], c1[0], N_segments+1)[1:-1]
    ys = torch.linspace(c0[1], c1[1], N_segments+1)[1:-1]
    init_points = torch.stack([xs, ys], dim=1)
    between_points = init_points.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([between_points], lr=lr)

    decoder_net.eval()
    for param in decoder_net.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = energy_piecewise(c0, c1, between_points, decoder_net, N_segments)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Data-space energy = {loss.item():.4f}")
    optimized_pts = torch.cat([c0.unsqueeze(0), between_points, c1.unsqueeze(0)], dim=0)
    return optimized_pts.detach().cpu().numpy()


def plot_latent_and_geodesic_piecewise(vae_model, latent_codes, labels, N_segments=10):
    plt.figure(figsize=(8, 8))
    #scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1], s=10, c=labels,
    #                      cmap="viridis", alpha=0.6, label="Latent codes")
    #plt.colorbar(scatter, label="Digit label")
    
    indices = np.random.choice(latent_codes.shape[0], size=2, replace=False)
    c0_np, c1_np = latent_codes[indices[0]], latent_codes[indices[1]]
    c0 = torch.tensor(c0_np, dtype=torch.float32)
    c1 = torch.tensor(c1_np, dtype=torch.float32)
    
    decoder_net = vae_model.decoder.decoder_net
    geod = compute_geodesic_piecewise(c0, c1, decoder_net, N_segments=N_segments, epochs=5000, lr=0.001)
    
    plt.plot(geod[:, 0], geod[:, 1], '-o', color='red', linewidth=2, label="Geodesic curve")
    plt.scatter([c0_np[0], c1_np[0]], [c0_np[1], c1_np[1]], color='black', s=50, zorder=5, label="Endpoints")

    plt.scatter(latent_codes[:, 0], latent_codes[:, 1], s=1, c=labels, alpha=0.5)
    plt.title("Geodesics in Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend()
    plt.savefig("figures/png/geod/geodesics_piecewise.png", dpi=300)
    plt.savefig("figures/svg/geod/geodesics_piecewise.svg", dpi=300)
    #plt.show()


def plot_latent_and_geodesic_polynomial(vae_model, latent_codes, labels, degree=4, N_points=50):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latent_codes[:, 0], latent_codes[:, 1], s=10, c=labels,
                          cmap="viridis", alpha=0.6, label="Latent codes")
    plt.colorbar(scatter, label="Digit label")
    
    indices = np.random.choice(latent_codes.shape[0], size=2, replace=False)
    c0_np, c1_np = latent_codes[indices[0]], latent_codes[indices[1]]
    c0 = torch.tensor(c0_np, dtype=torch.float32)
    c1 = torch.tensor(c1_np, dtype=torch.float32)
    
    decoder_net = vae_model.decoder.decoder_net
    geod, weights = compute_geodesic_polynomial(c0, c1, decoder_net, degree=degree, N_points=N_points, epochs=1000, lr=0.001)
    
    #plt.plot(geod[:, 0], geod[:, 1], '-o', color='red', linewidth=2, label="Poly geodesic")
    #plt.scatter([c0_np[0], c1_np[0]], [c0_np[1], c1_np[1]], color='black', s=50, zorder=5, label="Endpoints")
    #plt.title("Latent Space with Polynomial Geodesic")
    #plt.xlabel("z1")
    #plt.ylabel("z2")
    #plt.legend()
    #plt.savefig("geodesic_polynomial.png", dpi=300)
    #plt.show()

    polynomial_geodesics = []
    weights_tensor = torch.tensor(weights)
    t_values = torch.linspace(0.0, 1.0, 100)
    curve_points = torch.stack([polynomial_curve(t, c0, c1, weights_tensor) for t in t_values])
    polynomial_geodesics.append(curve_points.numpy())

    plt.figure(figsize=(8, 8))
    plt.scatter(latent_codes[:, 0], latent_codes[:, 1], s=1, c=labels, alpha=0.5)
    for i, geod in enumerate(polynomial_geodesics):
        plt.plot(geod[:, 0], geod[:, 1], '-', color="blue", label=f'Polynomial {i+1}')
    plt.title("Polynomial Geodesics in Latent Space")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.legend()
    #plt.show()
    plt.savefig("figures/png/geod/geodesic_polynomial.png", dpi=300)
    plt.savefig("figures/svg/geod/geodesic_polynomial.svg", dpi=300)