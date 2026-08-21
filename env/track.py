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

        # 4 Checkpoints to track progress. (x, y, width, height)
        self.checkpoints = [
            pygame.Rect(650, 295, 100, 10),  # Right
            pygame.Rect(400, 450, 10, 100),  # Bottom
            pygame.Rect(50, 295, 100, 10),  # Left
            pygame.Rect(400, 50, 10, 100),  # Top (Finish line)
        ]

    def draw(self, screen: pygame.Surface):
        screen.fill((34, 139, 34))  # Grass
        pygame.draw.ellipse(screen, (50, 50, 50), (50, 50, 700, 500))  # Track
        pygame.draw.ellipse(screen, (34, 139, 34), (150, 150, 500, 300))  # Inner grass

        for i, cp in enumerate(self.checkpoints):
            color = (255, 0, 0) if i == len(self.checkpoints) - 1 else (0, 0, 255)
            pygame.draw.rect(screen, color, cp, 2)
