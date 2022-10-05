import glob
import numpy as np
import cv2


class Data:
    def __init__(self, images_path, masks_path, resize_shape=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.resize_shape = resize_shape

        self.image_dataset = None
        self.mask_dataset = None

        self.data_preprocess = DataPreprocess()

    # Load Datasets
    def load_datasets(self):
        print('Loading dataset:')
        self.load_image_dataset()
        self.load_mask_dataset()

        self.image_dataset = self.data_preprocess.normalize_image_dataset(image_dataset=self.image_dataset)

    # Load Images
    def load_image_dataset(self):
        print('Loading images...')
        image_names = glob.glob(self.images_path + '*.png')
        image_names.sort()

        images = []

        for image_name in image_names:
            image = cv2.imread(image_name, cv2.IMREAD_COLOR)
            if self.resize_shape is not None:
                image = cv2.resize(image, self.image_shape, interpolation=cv2.INTER_NEAREST)
            images.append(image)

        self.image_dataset = np.array(images)

    # Load Masks
    def load_mask_dataset(self):
        print('Loading masks...')
        mask_names = glob.glob(self.masks_path + '*.png')
        mask_names.sort()

        masks = []

        for mask_name in mask_names:
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
            if self.resize_shape is not None:
                mask = cv2.resize(mask, self.image_shape, interpolation=cv2.INTER_NEAREST)
            masks.append(mask)
        masks_array = np.array(masks)
        self.mask_dataset = np.expand_dims(masks_array, axis=3)

    # Save Images
    def save_image_dataset(self, path):
        for i in range(len(self.image_dataset)):
            cv2.imwrite(f"{path}/image_{i}.png", self.image_dataset[i])

    # Save masks
    def save_mask_dataset(self, path):
        for i in range(len(self.mask_dataset)):
            cv2.imwrite(f"{path}/mask_{i}.png", self.mask_dataset[i])


class DataPreprocess:
    def __init__(self):
        # Mask for decreasing number of labels in masks
        self.mapping_labels = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 10: 9, 15: 10, 16: 11, 19: 12, 22: 13,
                               32: 14, 41: 15, 52: 16, 21: 17}

    # Function for decreasing number of labels in masks
    def decrease_label_nums(self, masks_dataset):
        new_mask_dataset = []
        for mask in masks_dataset:
            new_mask = self.update_labels(mask)
            new_mask_dataset.append(new_mask)
        return np.array(new_mask_dataset)

    def update_labels(self, mask):
        label_mask = np.zeros_like(mask)
        for k in self.mapping_labels:
            label_mask[mask == k] = self.mapping_labels[k]
        return label_mask

    # Function for normalizing Image Dataset
    def normalize_image_dataset(self, image_dataset):
        image_dataset = image_dataset / 255.

        return image_dataset
