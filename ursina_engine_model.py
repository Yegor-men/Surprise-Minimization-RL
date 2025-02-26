#%%
import torch

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
#%%
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, latent_size: int):
        super().__init__()

        self.latent = torch.zeros(latent_size)
        self.old_outputs = torch.zeros(1)

        self.encoder = nn.Sequential(
            nn.LazyLinear(out_features=latent_size * 2),
            nn.PReLU(),
            nn.LazyLinear(out_features=latent_size)
        )

    def update_hidden_states(self):
        pass

    def save_checkpoint(self):
        pass

    def load_checkpoint(self):
        pass

    def forward(self):
        pass