import random
from PIL import Image, ImageDraw
import math
import numpy as np

# Canvas settings
WIDTH, HEIGHT = 1500, 500
BACKGROUND = (0, 10, 50)

GRID_WIDTH, GRID_HEIGHT = 75, 25
BORDER = 3
SQUARE_SIZE = 14
GRID_SIZE = 20

MAX_JITTER = 5
MAX_ROTATION = 30


def cos_range(x):
    x = min(max(x, 0.0), 1.0)
    return 0.5 * (1 - math.cos(x * math.pi))


# Create the canvas
canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)

for i in range(0, GRID_WIDTH):
    for j in range(0, GRID_HEIGHT):

        # def rc(base, interp):
        #     c = random.random() * 255
        #     return min(max(int(base * interp + c * (1 - interp)), 0), 255)

        # # Random color with optional transparency
        # color = (
        #     rc(0, i/GRID_WIDTH),
        #     rc(150, i/GRID_WIDTH),
        #     rc(200, i/GRID_WIDTH),
        #     255,
        # )

        # def rc(base, range):
        #     p = 2 * random.random() - 1
        #     return min(max(int(base + p * range), 0), 255)
        # color = (
        #     rc(0, 0),
        #     rc(150, 0),
        #     rc(200, 50),
        #     255,
        # )

        color = np.array([0.0, 150, 200]) * (1 + 0.25 * (2 * random.random() - 1.0))
        color = tuple(np.clip(color.astype(int), 0, 255)) + (255,)

        # Draw square on its own transparent layer
        square = Image.new("RGBA", (SQUARE_SIZE, SQUARE_SIZE), (0, 0, 0, 0))
        draw = ImageDraw.Draw(square)
        draw.rectangle([0, 0, SQUARE_SIZE - 1, SQUARE_SIZE - 1], fill=color)

        # Random rotation
        angle = np.random.standard_normal() * MAX_ROTATION * cos_range(1 - i*1.1 / GRID_WIDTH) 
        rotated = square.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

        x = i * GRID_SIZE + BORDER + np.random.standard_normal() * MAX_JITTER * max(1 - i*1.05 / GRID_WIDTH, 0)
        y = j * GRID_SIZE + BORDER + np.random.standard_normal() * MAX_JITTER * max(1 - i*1.05 / GRID_WIDTH, 0)

        # Composite onto canvas
        canvas.paste(rotated, (int(x), int(y)), rotated)

# NUM_SQUARES = 40
# MIN_SIZE, MAX_SIZE = 30, 120

# # Create the canvas
# canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)

# for _ in range(NUM_SQUARES):
#     size = random.randint(MIN_SIZE, MAX_SIZE)

#     # Random color with optional transparency
#     color = (
#         random.randint(0, 255),
#         random.randint(0, 255),
#         random.randint(0, 255),
#         random.randint(120, 230),
#     )

#     # Draw square on its own transparent layer
#     square = Image.new("RGBA", (size, size), (0, 0, 0, 0))
#     draw = ImageDraw.Draw(square)
#     draw.rectangle([0, 0, size - 1, size - 1], fill=color)

#     # Random rotation
#     angle = random.uniform(0, 360)
#     rotated = square.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

#     # Random position
#     x = random.randint(-rotated.width // 2, WIDTH - rotated.width // 2)
#     y = random.randint(-rotated.height // 2, HEIGHT - rotated.height // 2)

#     # Composite onto canvas
#     canvas.paste(rotated, (x, y), rotated)

# Save to PNG without displaying
canvas.save("random_squares.png")
