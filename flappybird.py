import math
import os
import random
from random import randint

random.seed(0)
from collections import deque

import pygame
from pygame.locals import *

SIM_FPS = 30
ANIMATION_SPEED = 0.18
# ANIMATION_SPEED = 0.1
WIN_WIDTH = 284 * 2
WIN_HEIGHT = 512

def frames_to_msec(frames, fps=SIM_FPS):
    return 1000.0 * frames / fps

def msec_to_frames(milliseconds, fps=SIM_FPS):
    return fps * milliseconds / 1000.0

msec_in_one_frame = frames_to_msec(1)

class Bird(pygame.sprite.Sprite):
    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.18
    CLIMB_DURATION = 333.3

    def __init__(self, x, y, msec_to_climb, images):
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, velocity):
        self.y -= Bird.CLIMB_SPEED * msec_in_one_frame * velocity

        self.y = min(self.y, WIN_HEIGHT - Bird.HEIGHT)
        self.y = max(self.y, 0)

    @property
    def image(self):
        return self._img_wingup if pygame.time.get_ticks() % 500 >= 250 else self._img_wingdown

    @property
    def mask(self):
        return self._mask_wingup if pygame.time.get_ticks() % 500 >= 250 else self._mask_wingdown

    @property
    def rect(self):
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img, pipe_body_img):
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.convert()
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT -
             3 * Bird.HEIGHT -
             3 * PipePair.PIECE_HEIGHT) /
            PipePair.PIECE_HEIGHT
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i * PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        self.top_pieces += 1
        self.bottom_pieces += 1

        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self):
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self):
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self):
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self):
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self):
        self.x -= ANIMATION_SPEED * msec_in_one_frame

    def collides_with(self, bird):
        return pygame.sprite.collide_mask(self, bird)


def load_images():
    def load_image(img_file_name):
        base_dir = os.getcwd()
        file_name = os.path.join(base_dir, 'images', img_file_name)
        img = pygame.image.load(file_name)
        img.convert()
        return img

    return {'background': load_image('background.png'),
            'pipe-end': load_image('pipe_end.png'),
            'pipe-body': load_image('pipe_body.png'),
            'bird-wingup': load_image('bird_wing_up.png'),
            'bird-wingdown': load_image('bird_wing_down.png')}





def is_bird_between_pipes(bird, pipe):
    return False if bird.x + Bird.WIDTH < pipe.x or bird.x > pipe.x + PipePair.WIDTH else True


import numpy as np
import torch

torch.autograd.set_detect_anomaly(True)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


def get_screenshot():
    arr = pygame.surfarray.array3d(pygame.display.get_surface())
    return ((torch.from_numpy(arr).float().permute(2, 1, 0))/255).unsqueeze(0).to("cuda")


import flappybird_model
lat_size = 2048
model = flappybird_model.Model(
    hidden_size=lat_size,
    image_latent_size=lat_size,
).to("cuda")
decoder = flappybird_model.Decoder(
    hidden_size=lat_size,
    temperature=0.2
).to("cuda")

loss_fn = flappybird_model.RewardFunction()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
optimizer.zero_grad()

losses = []
vae_losses = []
s_factors = []
outputs = []
scores = [0]


def main():
    pygame.init()

    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption('Pygame Flappy Bird')

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)
    images = load_images()

    bird = Bird(
        x=50,
        y=int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2),
        msec_to_climb=2,
        images=(images['bird-wingup'], images['bird-wingdown'])
    )

    pipes = deque()

    frame_clock = 0
    done = paused = False

    is_touching_pipe, touched_pipe = False, False
    model_go = True

    curr_frame = 0

    while not done:
        clock.tick()
        curr_frame += 1

        if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images['pipe-end'], images['pipe-body'])
            pipes.append(pp)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                done = True
                break
            # elif event.type == KEYUP and event.key in (K_PAUSE, K_p):
            #     paused = not paused
            # elif event.type == MOUSEBUTTONUP or (event.type == KEYUP and event.key in (K_UP, K_RETURN, K_SPACE)):
            #     bird.msec_to_climb = Bird.CLIMB_DURATION

        # if paused:
        #     continue

        pipe_collision = any(p.collides_with(bird) for p in pipes)
        if pipe_collision:
            is_touching_pipe, touched_pipe = True, True
        else:
            is_touching_pipe = False

        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(images['background'], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        is_between_pipes = False
        for p in pipes:
            p.update()
            if is_bird_between_pipes(bird, p):
                is_between_pipes = True if not is_touching_pipe else False
            display_surface.blit(p.image, p.rect)

        screenshot = get_screenshot()

        vae_loss, euclid_dist, nh = model(screenshot, decoder.o_o)
        loss = loss_fn(vae_loss, euclid_dist, is_touching_pipe, is_between_pipes)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        no = decoder(nh)

        print(f"t: {curr_frame/30:.2f}", end=" | ")
        print(f"L {loss.item():.3f}", end=" | ")
        losses.append(loss.item())
        print(f"VAE L {vae_loss.item():,}", end=" | ")
        vae_losses.append(vae_loss.item())
        print(f"S factor {euclid_dist.item():.5f}", end=" | ")
        s_factors.append(euclid_dist.item())
        print(f"Out {no.squeeze().item():.2f}", end=" | ")
        outputs.append(no.squeeze().item())
        print(f"Between: {is_between_pipes}", end=" | ")
        print(f"Touching: {is_touching_pipe}", end=" | ")
        print()

        bird.update(velocity=round(no.squeeze().item())*2-1)
        display_surface.blit(bird.image, bird.rect)

        global scores
        for p in pipes:
            if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                p.score_counted = True
                if not touched_pipe:
                    scores[-1] += 1
                else:
                    scores.append(0)
                touched_pipe = False

        pygame.display.flip()
        frame_clock += 1

    pygame.quit()


main()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 2, figsize=(10, 15))

axes = axes.flatten()

axes[0].plot(losses)
axes[0].set_title('Loss')

# axes[1].plot(vae_losses)
# axes[1].set_title('VAE loss')

axes[2].plot(vae_losses)
axes[2].set_title('VAE Loss')

axes[3].plot(s_factors)
axes[3].set_title('S factor')

axes[4].plot(outputs)
axes[4].set_title('Output')

axes[5].plot(scores)
axes[5].set_title('Scores')

plt.tight_layout()

plt.show()
