import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class Generator(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def generate_faults(
    data, threshold: float = 0.8, epochs: int = 50, batch_size: int = 32
):
    """Train a simple GAN and return generated fault samples."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.as_tensor(data, dtype=torch.float32, device=device)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    noise_dim = data.shape[1]
    gen = Generator(noise_dim, noise_dim).to(device)
    disc = Discriminator(noise_dim).to(device)

    optim_g = torch.optim.Adam(gen.parameters(), lr=1e-3)
    optim_d = torch.optim.Adam(disc.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            z = torch.randn(len(batch), noise_dim, device=device)
            fake = gen(z)

            # Train discriminator
            optim_d.zero_grad()
            pred_real = disc(batch)
            pred_fake = disc(fake.detach())
            loss_d = bce(pred_real, torch.ones_like(pred_real)) + bce(
                pred_fake, torch.zeros_like(pred_fake)
            )
            loss_d.backward()
            optim_d.step()

            # Train generator
            optim_g.zero_grad()
            pred_fake = disc(fake)
            loss_g = bce(pred_fake, torch.ones_like(pred_fake))
            loss_g.backward()
            optim_g.step()

    with torch.no_grad():
        z = torch.randn(len(data), noise_dim, device=device)
        generated = gen(z).cpu().numpy()

    fault_mask = (
        disc(torch.as_tensor(generated, dtype=torch.float32, device=device))
        .cpu()
        .numpy()
        .flatten()
        > threshold
    )
    return generated[fault_mask]
