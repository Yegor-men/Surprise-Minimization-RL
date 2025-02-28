
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, latent_size):
        """ In size is [3, 512, 568] """
        super().__init__()

        self.latent = torch.zeros(latent_size)
        self.prev_output = torch.zeros(1)

        self.latent_norm = nn.LayerNorm(latent_size)
        self.pred_lat_norm = nn.LayerNorm(latent_size)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=0),
            nn.Flatten()
        )

        self.latent_updater = nn.Sequential(
            nn.LazyLinear(out_features=latent_size * 2),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=latent_size)
        )

        self.latent_predictor = nn.Sequential(
            nn.LazyLinear(out_features=latent_size * 2),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=latent_size)
        )

        self.decoder = nn.Sequential(
            nn.LazyLinear(out_features=1),
            nn.Sigmoid()
        )

    def update_hidden_values(self, new_latent, output):
        self.latent = new_latent
        self.prev_output = output

    def forward(self, image, is_touching):
        self.latent = self.latent.detach()
        self.prev_output = self.prev_output.detach()

        image_latent = self.image_encoder(image).squeeze()
        temp1 = torch.cat([image_latent, is_touching], dim=0)
        new_latent = self.latent_norm(self.latent + self.latent_updater(temp1))

        temp2 = torch.cat([self.latent, self.prev_output])
        pred_latent = self.pred_lat_norm(self.latent + self.latent_predictor(temp2))

        outputs = self.decoder(new_latent)

        return new_latent, pred_latent, outputs, is_touching

import torch
from torch import nn


class RewardFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, new_latent, predicted_latent, is_touching):
        euclid_dist = torch.cdist(new_latent.unsqueeze(0), predicted_latent.unsqueeze(0))

        reward = -euclid_dist - is_touching
        loss = torch.exp(-reward)
        return loss, euclid_dist