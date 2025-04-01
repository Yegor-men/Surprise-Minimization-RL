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
        self.queue = deque([torch.zeros(self.input_shape) for _ in range(max_len)], maxlen=max_len)

    def return_queue_as_batched_tensor(self):
        return torch.stack(list(self.queue), dim=0)

    def get_oldest(self):
        return self.queue[0]

    def add_element(self, element: torch.tensor):
        self.queue.append(element)


class ReinforceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logprob, is_touching: bool, reconstruction_loss):
        reward = -1 if is_touching else 1
        loss = reward * (-logprob) + reconstruction_loss
        return loss


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
            input_shape: tuple,
    ):
        super().__init__()

        self.image_history = FixedSizeQueue(image_history_length, input_shape=input_shape)

        self.h = torch.zeros(1, h_c_size)
        self.c = torch.zeros(1, h_c_size)
        self.lstm = nn.LSTM(input_size=compressed_image_size, hidden_size=h_c_size)

        reconstructed_image_c = reconstructed_image_size[0]
        reconstructed_image_h = reconstructed_image_size[1]
        reconstructed_image_w = reconstructed_image_size[2]

        self.resolution_resizer = ResolutionResizer(reconstructed_image_h, reconstructed_image_w)

        reconstructed_image_num_pixels = reconstructed_image_c * reconstructed_image_h * reconstructed_image_w

        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=compressed_image_size),
        )

        self.image_reconstructor = nn.Sequential(
            nn.LazyLinear(out_features=reconstructed_image_num_pixels),
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
        self.image_history.add_element(new_image.squeeze())
        images = self.image_history.return_queue_as_batched_tensor()
        small_images = self.resolution_resizer(images).detach()
        compressed_images = self.image_encoder(images)
        temp_h, temp_c = self.h, self.c
        for i in range(compressed_images.size(0)):
            _, (temp_h, temp_c) = self.lstm(compressed_images[i].unsqueeze(0), (temp_h, temp_c))
        distribution_logit = self.choice_maker(temp_h)
        distribution = dist.Bernoulli(logits=distribution_logit)
        choice = distribution.sample()
        logprob = distribution.log_prob(choice)

        reconstructed_images = self.image_reconstructor(compressed_images)
        grid = make_grid(reconstructed_images, nrow=4)
        numpy_grid = grid.permute(1, 2, 0).cpu().detach().numpy()
        cv2.imshow("VAE reconstruction", numpy_grid)
        cv2.waitKey(1)

        reconstruction_loss = F.mse_loss(reconstructed_images, small_images, reduction='sum')

        return choice, logprob, reconstruction_loss
