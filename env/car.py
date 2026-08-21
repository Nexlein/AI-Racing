import math

import numpy as np
import pygame


class Car:
    def __init__(self, pos, angle):
        self.pos = list(pos)
        self.angle = angle
        self.speed = 0
        self.max_speed = 10
        self.acceleration = 0.5
        self.friction = 0.05
        self.turn_speed = 5
        self.length, self.width = 40, 20
        self.surface = pygame.Surface((self.length, self.width), pygame.SRCALPHA)
        self.surface.fill((200, 50, 50))
        self.radars = []

    def step(self, throttle, steering):
        self.speed += throttle * self.acceleration
        self.speed *= 1 - self.friction
        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))

        if abs(self.speed) > 0.1:
            direction = 1 if self.speed > 0 else -1
            self.angle += steering * self.turn_speed * direction

        rad = math.radians(self.angle)
        self.pos[0] += math.cos(rad) * self.speed
        self.pos[1] -= math.sin(rad) * self.speed

    def check_collision(self, track_mask):
        car_image = pygame.transform.rotate(self.surface, self.angle)
        offset = (
            int(self.pos[0] - car_image.get_width() / 2),
            int(self.pos[1] - car_image.get_height() / 2),
        )
        car_mask = pygame.mask.from_surface(car_image)
        return track_mask.overlap(car_mask, offset) is not None

    def get_raycasts(self, track_mask, num_rays=5, ray_len=200):
        self.radars = []
        distances = []
        angles = np.linspace(-90, 90, num_rays)

        for d_angle in angles:
            ray_angle = self.angle + d_angle
            rad = math.radians(ray_angle)
            for dist in range(1, ray_len, 5):
                x = int(self.pos[0] + math.cos(rad) * dist)
                y = int(self.pos[1] - math.sin(rad) * dist)

                if (
                    x < 0
                    or y < 0
                    or x >= track_mask.get_size()[0]
                    or y >= track_mask.get_size()[1]
                ):
                    distances.append(dist / ray_len)
                    self.radars.append((x, y))
                    break
                if track_mask.get_at((x, y)):
                    distances.append(dist / ray_len)
                    self.radars.append((x, y))
                    break
            else:
                distances.append(1.0)
                self.radars.append(
                    (
                        int(self.pos[0] + math.cos(rad) * ray_len),
                        int(self.pos[1] - math.sin(rad) * ray_len),
                    )
                )
        return np.array(distances, dtype=np.float32)

    def draw(self, screen):
        car_image = pygame.transform.rotate(self.surface, self.angle)
        rect = car_image.get_rect(center=self.pos)
        screen.blit(car_image, rect.topleft)
        for radar in self.radars:
            pygame.draw.line(screen, (0, 255, 0), self.pos, radar, 1)
            pygame.draw.circle(screen, (0, 255, 0), radar, 3)
