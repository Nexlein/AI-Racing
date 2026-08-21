import pygame


class Track:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

        self.image = pygame.Surface((width, height), pygame.SRCALPHA)
        self.image.fill((255, 255, 255, 255))  # Solid (collision)
        pygame.draw.ellipse(self.image, (0, 0, 0, 0), (50, 50, 700, 500))  # Track path
        pygame.draw.ellipse(
            self.image, (255, 255, 255, 255), (150, 150, 500, 300)
        )  # Inner solid

        self.mask = pygame.mask.from_surface(self.image)
        self.start_pos = [400, 100]
        self.start_angle = 0

    def draw(self, screen):
        screen.fill((34, 139, 34))  # Grass
        pygame.draw.ellipse(screen, (50, 50, 50), (50, 50, 700, 500))  # Track
        pygame.draw.ellipse(screen, (34, 139, 34), (150, 150, 500, 300))  # Inner grass
