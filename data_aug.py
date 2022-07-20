import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm  # Progress bar
import imageio
from albumentations import HorizontalFlip, VerticalFlip, Rotate
import matplotlib.pyplot as plt

# Hyperparameters
SIZE = (512, 512)
DATA_PATH = "data/"
NEW_DATA_PATH_TRAIN = "new_data/train/"
NEW_DATA_PATH_TEST = "new_data/test/"


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    train_x = sorted(glob(os.path.join(path, "training", "images", "*.tif")))
    train_y = sorted(glob(os.path.join(path, "training", "mask", "*.gif")))

    test_x = sorted(glob(os.path.join(path, "test", "images", "*.tif")))
    test_y = sorted(glob(os.path.join(path, "test", "mask", "*.gif")))

    return (train_x, train_y), (test_x, test_y)


def augment_data(images, masks, save_path, augment=True):
    for idx, (data, target) in tqdm(enumerate(zip(images, masks)), total=len(images)):
        name = data.split("/")[-1].split(".")[0]

        data: np.ndarray = plt.imread(data)
        target: np.ndarray = imageio.mimread(target)[0]

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=data, mask=target)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=data, mask=target)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=data, mask=target)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            X = [data, x1, x2, x3]
            Y = [target, y1, y2, y3]

        else:
            X = [data]
            Y = [target]

        index = 0
        for augmented_x, augmented_y in zip(X, Y):
            augmented_x = cv2.resize(augmented_x, SIZE)
            augmented_y = cv2.resize(augmented_y, SIZE)

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", tmp_image_name)
            mask_path = os.path.join(save_path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, augmented_x)
            cv2.imwrite(mask_path, augmented_y)

            index += 1


if __name__ == "__main__":
    np.random.seed(42)
    (train_X, train_Y), (test_X, test_Y) = load_data(DATA_PATH)

    print(f"Train: {len(train_X)} - {len(train_Y)}")
    print(f"Test: {len(test_X)} - {len(test_Y)}")

    create_dir(NEW_DATA_PATH_TRAIN + "image/")
    create_dir(NEW_DATA_PATH_TRAIN + "mask/")
    create_dir(NEW_DATA_PATH_TEST + "image/")
    create_dir(NEW_DATA_PATH_TEST + "mask/")

    augment_data(train_X, train_Y, NEW_DATA_PATH_TRAIN, augment=True)
    augment_data(test_X, test_Y, NEW_DATA_PATH_TEST, augment=False)
