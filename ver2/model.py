import math

import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as dist
from torchvision.utils import make_grid

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from collections import deque

import cv2


class FixedSizeQueue:
    def __init__(self, max_len: int, input_shape: tuple):
        self.input_shape = input_shape
        self.queue = deque([torch.zeros(self.input_shape).to("cuda") for _ in range(max_len)], maxlen=max_len)

    def return_queue_as_batched_tensor(self):
        return torch.stack(list(self.queue), dim=0)

    def get_oldest(self):
        return self.queue[0]

    def add_element(self, element: torch.tensor):
        self.queue.append(element)


class ReinforceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logprob, distribution_entropy, is_touching: bool, is_between: bool):
        reward = -1 if is_touching else (1 if is_between else 0)
        loss = reward * (-logprob) - 0.001 * distribution_entropy
        return loss, reward


class ResolutionResizer(nn.Module):
    def __init__(self, target_height, target_width, mode='interp'):
        super().__init__()
        self.interp_mode = "bicubic"
        self.target_height = target_height
        self.target_width = target_width
        self.mode = mode

    def forward(self, x):
        if self.mode == 'avg':
            return F.adaptive_avg_pool2d(x, (self.target_height, self.target_width))
        elif self.mode == 'interp':
            return F.interpolate(x, size=(self.target_height, self.target_width), mode=self.interp_mode,
                                 align_corners=False)


class Model(nn.Module):
    def __init__(
            self,
            image_history_length: int,
            h_c_size: int,
            compressed_image_size: int,
            reconstructed_image_size: tuple,
            reshape_image_to: tuple,
    ):
        super().__init__()

        self.image_history = FixedSizeQueue(image_history_length, input_shape=reshape_image_to)
        self.small_image_history = FixedSizeQueue(image_history_length, input_shape=reconstructed_image_size)

        self.h = torch.zeros(1, h_c_size).to("cuda")
        self.c = torch.zeros(1, h_c_size).to("cuda")
        self.lstm = nn.LSTM(input_size=compressed_image_size, hidden_size=h_c_size)

        reconstructed_image_c = reconstructed_image_size[0]
        reconstructed_image_h = reconstructed_image_size[1]
        reconstructed_image_w = reconstructed_image_size[2]

        self.resize_image = ResolutionResizer(reshape_image_to[1], reshape_image_to[2])
        self.smallen_image = ResolutionResizer(reconstructed_image_h, reconstructed_image_w)

        reconstructed_image_num_pixels = reconstructed_image_c * reconstructed_image_h * reconstructed_image_w

        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=compressed_image_size),
            nn.LayerNorm(compressed_image_size),
        )

        self.image_reconstructor = nn.Sequential(
            nn.LazyLinear(out_features=(reconstructed_image_num_pixels + compressed_image_size) // 2),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=reconstructed_image_num_pixels),
            nn.Sigmoid(),
            nn.Unflatten(1, (reconstructed_image_c, reconstructed_image_h, reconstructed_image_w)),
        )

        self.choice_maker = nn.Sequential(
            nn.LazyLinear(out_features=1)
        )

    def update_hidden_states(self):
        oldest_image = self.image_history.get_oldest().unsqueeze(0)
        with torch.no_grad():
            compressed_image = self.image_encoder(oldest_image)
            _, (n_h, n_c) = self.lstm(compressed_image, (self.h, self.c))
            self.h, self.c = n_h.detach(), n_c.detach()

    def forward(self, new_image):
        resized_image = self.resize_image(new_image)
        small_image = self.smallen_image(new_image)
        self.image_history.add_element(resized_image.squeeze())
        self.small_image_history.add_element(small_image.squeeze())
        images = self.image_history.return_queue_as_batched_tensor()
        small_images = self.small_image_history.return_queue_as_batched_tensor()

        compressed_images = self.image_encoder(images)
        temp_h, temp_c = self.h, self.c
        for i in range(compressed_images.size(0)):
            _, (temp_h, temp_c) = self.lstm(compressed_images[i].unsqueeze(0), (temp_h, temp_c))
        distribution_logit = self.choice_maker(temp_h)

        distribution = dist.Bernoulli(logits=distribution_logit)
        choice = distribution.sample()
        logprob = distribution.log_prob(choice)
        distribution_entropy = distribution.entropy()

        reconstructed_images = self.image_reconstructor(compressed_images.detach().clone())

        recon_grid = make_grid(
            reconstructed_images,
            nrow=int(math.sqrt(len(self.image_history.queue)))
        )
        small_grid = make_grid(
            small_images,
            nrow=int(math.sqrt(len(self.image_history.queue)))
        )
        combined_grid = torch.cat((small_grid, recon_grid), dim=2)
        numpy_grid = combined_grid.permute(1, 2, 0).cpu().detach().numpy()
        cv2.imshow("Autoencoder reconstruction", numpy_grid)
        cv2.waitKey(1)

        reconstruction_loss = F.mse_loss(reconstructed_images, small_images, reduction='sum')

        return choice, logprob, distribution_entropy, reconstruction_loss
