# Most of this code was given by the instructor
import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Generate a dataset of images using the specified font

# Load the font and set the font size to 42
font = ImageFont.truetype(
    "Habbakuk.TTF", 42
)  # FIXME: only works in src folder right now, fix this

# Set kernel
kernel = np.ones((5, 5), np.uint8)

# Character mapping for each of the 27 tokens
char_map = {
    "Alef": ")",
    "Ayin": "(",
    "Bet": "b",
    "Dalet": "d",
    "Gimel": "g",
    "He": "x",
    "Het": "h",
    "Kaf": "k",
    "Kaf-final": "\\",
    "Lamed": "l",
    "Mem": "{",
    "Mem-medial": "m",
    "Nun-final": "}",
    "Nun-medial": "n",
    "Pe": "p",
    "Pe-final": "v",
    "Qof": "q",
    "Resh": "r",
    "Samekh": "s",
    "Shin": "$",
    "Taw": "t",
    "Tet": "+",
    "Tsadi-final": "j",
    "Tsadi-medial": "c",
    "Waw": "w",
    "Yod": "y",
    "Zayin": "z",
}

# Returns a grayscale image based on specified label of img_size
def create_image(label, img_size):
    if label not in char_map:
        raise KeyError("Unknown label!")

    # Create blank image and create a draw interface
    img = Image.new("L", img_size, 255)
    draw = ImageDraw.Draw(img)

    # Get size of the font and draw the token in the center of the blank image
    w, h = font.getsize(char_map[label])
    draw.text(((img_size[0] - w) / 2, (img_size[1] - h) / 2), char_map[label], 0, font)

    return np.array(img)


# Define the transformation pipeline, including:
# - Brightness
# - Elastic transform
# - Affine transform
# - Gaussian blur
# TODO: add dilute and erode transformations
transform = A.Compose(
    [
        A.RandomBrightnessContrast(p=0.5),
        A.ElasticTransform(p=0.5, alpha=10, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        A.Affine(
            rotate=(-10, 10), shear=(-0.5, 0.5), scale=(0.9, 1.2), cval=255, p=0.5
        ),
        A.GaussianBlur(p=0.5),
    ]
)

# Create a folder to store the images, generate the images with transforms for every character and save them to their respective folders
for name in char_map.keys():  # the dictionary
    img = create_image(name, (50, 50))  # create empty image
    for i in range(50):  # TODO See if change
        transformed = transform(image=img)["image"]  # apply a random transform
        if not Path.exists(
            Path(Path("new_data") / name)
        ):  # create a folder for the character
            os.makedirs(f"new_data/{name}")
        Image.fromarray(transformed).save(f"new_data/{name}/{str(i)}.png")
