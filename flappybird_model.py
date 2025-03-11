import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_reconstruction, mu, logvar):
        recon_loss = F.mse_loss(x_reconstruction, x, reduction='sum')  # or mean
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss


class Decoder(nn.Module):
    def __init__(self, hidden_size, temperature):
        super().__init__()

        self.temperature = temperature

        self.o_o = torch.zeros(1, 1).to("cuda")

        self.decoder = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=1)
        )

    def forward(self, n_h):
        clone_n_h = n_h.detach()
        output = self.decoder(clone_n_h)
        dist = torch.distributions.RelaxedBernoulli(temperature=self.temperature, logits=output)
        n_o = dist.rsample()
        self.o_o = n_o
        return n_o


import numpy as np
import cv2

class CNNOneToOne(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size -1,
            dilation=dilation,
        )

    def forward(self, image):
        temp = self.activation(self.cnn(image))
        residual = temp + image
        return residual

class CNNHalver(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size//2)-1,
        )

    def forward(self, image):
        return self.activation(self.cnn(image))

class CNNDoubler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.activation = nn.LeakyReLU()
        self.cnn = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=(kernel_size//2)-1,
        )
    def forward(self, image):
        return self.activation(self.cnn(image))

class Model(nn.Module):
    def __init__(self, hidden_size, image_latent_size):
        super().__init__()

        self.h = torch.zeros(1, hidden_size).to("cuda")
        self.c = torch.zeros(1, hidden_size).to("cuda")

        self.vae_enc = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, stride=2, padding=0),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=0),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2, stride=2, padding=0),
            CNNHalver(in_channels=3, out_channels=4, kernel_size=4),
            CNNHalver(in_channels=4, out_channels=8, kernel_size=4),
            CNNHalver(in_channels=8, out_channels=4, kernel_size=4),
            nn.Flatten()
        )
        self.fc_mu = nn.LazyLinear(image_latent_size)
        self.fc_logvar = nn.LazyLinear(image_latent_size)
        self.z_fc = nn.LazyLinear(4 * int(512 / 8) * int(568 / 8))
        self.vae_dec = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=4, out_channels=8, kernel_size=2, stride=2, padding=0),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=2, stride=2, padding=0),
            # nn.LeakyReLU(),
            # nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=2, stride=2, padding=0),
            CNNDoubler(in_channels=4, out_channels=8, kernel_size=4),
            CNNDoubler(in_channels=8, out_channels=4, kernel_size=4),
            CNNDoubler(in_channels=4, out_channels=3, kernel_size=4),
            nn.Sigmoid()
        )
        self.vae_loss = VAELoss()

        self.lstm_main = nn.LSTMCell(input_size=image_latent_size, hidden_size=hidden_size)
        self.lstm_pred = nn.LSTMCell(input_size=1, hidden_size=hidden_size)

    def forward(self, image, o_o):
        self.h = self.h.detach()
        self.c = self.c.detach()

        encoded_image = self.vae_enc(image)
        mu = self.fc_mu(encoded_image)
        logvar = self.fc_logvar(encoded_image)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        z_fc = self.z_fc(z)
        z_reshaped = torch.reshape(z_fc, (1, 4, 64, 71))
        reconstructed_image = self.vae_dec(z_reshaped)

        numpy_image = reconstructed_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        cv2.imshow("VAE reconstruction", numpy_image)
        cv2.waitKey(1)

        vae_loss = self.vae_loss(image, reconstructed_image, mu, logvar)

        n_h, n_c = self.lstm_main(z, (self.h, self.c))
        p_h, _ = self.lstm_pred(o_o, (self.h, self.c))

        euclid_dist = torch.cdist(n_h, p_h)

        return vae_loss, euclid_dist.squeeze(), n_h


class RewardFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, vae_loss, euclid_dist, is_touching, is_between):
        reward_value = -float(is_touching) * 10 + float(is_between) * 10
        reward = torch.tensor([reward_value])
        exp_reward = torch.exp(-reward).to("cuda")
        loss = vae_loss + euclid_dist + exp_reward
        return loss
