import torch

def reparameterize(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(mu)
    z = mu + torch.exp(0.5 * logvar) * noise
    return z

def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    kl = - 0.5 * (1 + logvar - mu.pow(2) - torch.exp(logvar))
    kl = kl.reshape(mu.shape[0], -1)
    kl = kl.sum(dim = 1)
    kl = kl.mean()
    return kl

def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    kl_loss = kl_divergence(mu, logvar)
    reconstruction_loss = (reconstruction - target).pow(2).reshape(reconstruction.shape[0], -1).sum(dim = 1).mean()
    loss = reconstruction_loss + kl_weight * kl_loss
    return loss, reconstruction_loss, kl_loss

class VAEEncoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.features = torch.nn.Sequential(
            # [B, 3, 32, 32]
            torch.nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size = 4,
                stride = 2,
                padding=1,
            ),
            # [B, 64, 16, 16]
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                base_channels,
                2 * base_channels,
                kernel_size = 4,
                stride= 2,
                padding= 1,
            ),
            # [B, 128, 8, 8]
            torch.nn.SiLU(),
            torch.nn.Conv2d(
                2 * base_channels,
                2 * base_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            torch.nn.SiLU()
        )
        self.to_mu = torch.nn.Conv2d(
            2 * base_channels,
            latent_channels, 
            kernel_size=3,
            padding=1
        )
        self.to_logvar = torch.nn.Conv2d(
            2 * base_channels,
            latent_channels,
            kernel_size=3,
            padding=1
        )
    def forward(self, x):
        latent = self.features(x)
        mu = self.to_mu(latent)
        logvar = self.to_logvar(latent)
        return mu, logvar

class VAEDecoder(torch.nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()
        self.net = torch.nn.Sequential(
            # [B, 4, 8, 8]
            torch.nn.Conv2d(
                latent_channels,
                2 * base_channels,
                kernel_size=3, 
                padding=1,
            ),
            # [B, 128, 8, 8]
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(
                2 * base_channels,
                base_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            # [B, 64, 16, 16]
            torch.nn.SiLU(),
            torch.nn.ConvTranspose2d(
                base_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            # [B, 3, 32, 32]
            torch.nn.Tanh(),
        )
    def forward(self,z):
        return self.net(z)

class VAE(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        latent_channels: int = 4,
    ):
        super().__init__()

        self.encoder = VAEEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

        self.decoder = VAEDecoder(
            out_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
        )

    def encode(
        self,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar, noise)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar, noise)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar, z

def vae_train_step(
    model: VAE,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    kl_weight: float = 1.0,
) -> tuple[float, float, float]:
    reconstruction, mu, logvar, z = model(images)
    loss, reconstruction_loss, kl_loss = vae_loss(reconstruction, images, mu, logvar, kl_weight)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return (
        loss.item(),
        reconstruction_loss.item(),
        kl_loss.item()
    )