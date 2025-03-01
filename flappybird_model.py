import torch
from torch import nn

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from collections import deque


class FixedSizeQueue:
    def __init__(self, max_length: int, hidden_size: int, output_size: int, input_size: tuple[int, int, int]):
        c, h, w = input_size
        s_input = torch.zeros([c, h, w])
        s_h = torch.zeros(1, hidden_size)
        s_c = torch.zeros(1, hidden_size)
        s_ph = torch.zeros(1, hidden_size)
        s_out = torch.zeros(1, output_size)

        self.queue = deque([(
            s_input.clone(),
            s_h.clone(),
            s_c.clone(),
            s_ph.clone(),
            s_out.clone()
        ) for _ in range(max_length)], maxlen=max_length)

    def enqueue(self, input, new_h, new_c, pred_h, output):
        self.queue.append((input, new_h, new_c, pred_h, output))
        self.detach_oldest()

    def detach_oldest(self):
        oldest_item = self.queue[0]
        self.queue[0] = tuple(tensor.detach() for tensor in oldest_item)

    def get_newest(self):
        i, h, c, ph, o = self.queue[-1]
        return i, h, c, ph, o


class Model(nn.Module):
    def __init__(self, hidden_size, noise, memory_length, image_latent_size):
        """ Image size is [3, 512, 568] """
        super().__init__()

        self.memory = FixedSizeQueue(
            max_length=memory_length,
            hidden_size=hidden_size,
            output_size=1,
            input_size=(3, 512, 568)
        )

        self.noise = noise

        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=0),
            nn.Flatten(),
            nn.LazyLinear(out_features=image_latent_size),
            nn.LeakyReLU()
        )

        self.LSTMCell = nn.LSTMCell(input_size=image_latent_size, hidden_size=hidden_size)

        self.predict_h = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size * 2),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=hidden_size)
        )

        self.decoder = nn.Sequential(
            nn.LazyLinear(out_features=hidden_size),
            nn.LeakyReLU(),
            nn.LazyLinear(out_features=1)
        )

    def forward(self, image):
        o_i, o_h, o_c, o_ph, o_o = self.memory.get_newest()

        image_latent = self.image_encoder(image)
        # temp1 = torch.cat([image_latent, is_touching], dim=0)
        n_h, n_c = self.LSTMCell(image_latent, (o_h, o_c))

        temp2 = torch.cat([o_h, o_o], dim=1)
        n_ph = self.predict_h(temp2)

        outputs = self.decoder(n_h)
        dist = torch.distributions.RelaxedBernoulli(temperature=self.noise, logits=outputs)
        n_o = dist.rsample()

        self.memory.enqueue(image, n_h, n_c, n_ph, n_o)

        return image, n_h, n_c, n_ph, n_o


class RewardFunction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, memory_entry, is_touching, is_between):
        i, h, c, ph, o = memory_entry

        euclid_dist = torch.cdist(h, ph)
        between = torch.ones(1, 1) if is_between else torch.zeros(1, 1)
        reward = -euclid_dist - is_touching + between
        loss = torch.exp(-reward)
        return loss, euclid_dist
