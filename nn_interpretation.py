import matplotlib.pyplot as plt
from data import *

from tensorflow.keras.models import load_model


class NNInterpretation:
    def __init__(self, data, model_path):
        self.data = data
        self.model = load_model(model_path)

    # Show: Original Image, Predicted Mask, True Mask
    def compare_predictions_with_masks(self):
        dataset_length = len(self.data.image_dataset)
        for i in range(dataset_length):
            id = np.random.randint(dataset_length)
            rand_img = self.data.image_dataset[id][np.newaxis, ...]
            pred_mask = self.model.predict(rand_img)[0]
            pred_mask = np.argmax(pred_mask, axis=2)
            true_mask = self.data.mask_dataset[id]

            plt.subplot(1, 3, 1)
            plt.imshow(rand_img[0])
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(pred_mask)
            plt.title("Predicted Mask")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(true_mask)
            plt.title("True Mask")
            plt.axis('off')

            plt.tight_layout()
            plt.show()


if __name__ == '__main__':
    images_path = "ade20k-4379-outdoor-images/images/"
    masks_path = "ade20k-4379-outdoor-images/masks/"

    data = Data(images_path, masks_path)
    data.load_datasets()

    model_path = 'UNet-Trained-Model-best.h5'

    nn_interpretation = NNInterpretation(data=data, model_path=model_path)
    nn_interpretation.compare_predictions_with_masks()
