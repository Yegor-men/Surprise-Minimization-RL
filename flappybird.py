import math
import os
from random import randint
from collections import deque

import pygame
from pygame.locals import *

FPS = 60
ANIMATION_SPEED = 0.18
WIN_WIDTH = 284 * 2
WIN_HEIGHT = 512


class Bird(pygame.sprite.Sprite):
    WIDTH = HEIGHT = 32
    SINK_SPEED = 0.18
    CLIMB_SPEED = 0.3
    CLIMB_DURATION = 333.3

    def __init__(self, x, y, msec_to_climb, images):
        super(Bird, self).__init__()
        self.x, self.y = x, y
        self.msec_to_climb = msec_to_climb
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def update(self, delta_frames=1):
        if self.msec_to_climb > 0:
            frac_climb_done = 1 - self.msec_to_climb / Bird.CLIMB_DURATION
            self.y -= (Bird.CLIMB_SPEED * frames_to_msec(delta_frames) *
                       (1 - math.cos(frac_climb_done * math.pi)))
            self.msec_to_climb -= frames_to_msec(delta_frames)
        else:
            self.y += Bird.SINK_SPEED * frames_to_msec(delta_frames)

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

    def update(self, delta_frames=1):
        self.x -= ANIMATION_SPEED * frames_to_msec(delta_frames)

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


def frames_to_msec(frames, fps=FPS):
    return 1000.0 * frames / fps


def msec_to_frames(milliseconds, fps=FPS):
    return fps * milliseconds / 1000.0


import numpy as np
import torch


def get_screenshot():
    surface = pygame.display.get_surface()
    arr = pygame.surfarray.array3d(surface)
    tensor = torch.from_numpy(arr).float()
    tensor = tensor.permute(2, 0, 1)
    return tensor


import flappybird_model

model = flappybird_model.Model(256)
loss_fn = flappybird_model.RewardFunction()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()
optimizer.zero_grad()

losses = []
s_factors = []
outputs = []
scores = [0]


def main():
    global losses
    global s_factors
    global outputs
    global scores

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

    touching_pipe, touched_pipe = False, False

    while not done:
        clock.tick(FPS)

        if not (paused or frame_clock % msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images['pipe-end'], images['pipe-body'])
            pipes.append(pp)

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYUP and event.key == K_ESCAPE):
                done = True
                break
            elif event.type == KEYUP and event.key in (K_PAUSE, K_p):
                paused = not paused
            elif event.type == MOUSEBUTTONUP or (event.type == KEYUP and event.key in (K_UP, K_RETURN, K_SPACE)):
                bird.msec_to_climb = Bird.CLIMB_DURATION

        if paused:
            continue

        pipe_collision = any(p.collides_with(bird) for p in pipes)
        if pipe_collision:
            touching_pipe, touched_pipe = True, True
        else:
            touching_pipe = False

        screenshot = get_screenshot()
        touching = torch.ones(1) if touching_pipe else torch.zeros(1)

        new_lat, pred_lat, output, is_touch = model(screenshot, touching)
        loss, s_factor = loss_fn(new_lat, pred_lat, is_touch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.update_hidden_values(
            new_latent=new_lat,
            output=output
        )
        print(f"L {loss.item():.3f}", end=" | ")
        losses.append(loss.item())
        print(f"S factor {s_factor.item():.3f}", end=" | ")
        s_factors.append(s_factor.item())
        print(f"Out {output.item():.5f}", end=" | ")
        outputs.append(output.item())
        print()
        if round(output.item()) == 1:
            bird.msec_to_climb = Bird.CLIMB_DURATION

        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(images['background'], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            p.update()
            display_surface.blit(p.image, p.rect)

        bird.update()
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

        # score_surface = score_font.render(str(score), True, (255, 255, 255))
        # score_x = WIN_WIDTH / 2 - score_surface.get_width() / 2
        # display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        frame_clock += 1

    # print(f"Game over! Scores: {scores}")
    pygame.quit()


main()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes = axes.flatten()

axes[0].plot(losses)
axes[0].set_title('Loss')

axes[1].plot(s_factors)
axes[1].set_title('S factor')

axes[2].plot(outputs)
axes[2].set_title('Output')

axes[3].plot(scores)
axes[3].set_title('Scores')

# axes[4].plot(loss_history)
# axes[4].set_title('Loss (lower is better)')
#
# axes[5].plot(surprise_factor_history)
# axes[5].set_title('Surprise Factor')
#
# axes[6].plot(battery_hist)
# axes[6].set_title('Battery')
#
# axes[7].plot(v_reward_history, label="Velocity")
# axes[7].plot(s_reward_history, label="S factor")
# axes[7].plot(total_reward_history, label="Total")
# axes[7].set_title('Reward')
# axes[7].legend()

plt.tight_layout()

plt.show()
