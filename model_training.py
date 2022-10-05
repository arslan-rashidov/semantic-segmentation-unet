from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split

from data import *

import segmentation_models as sm

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.metrics import MeanIoU, OneHotMeanIoU


class ModelTraining:
    def __init__(self, data, n_classes, image_shape):
        self.data = data
        self.n_classes = n_classes
        self.image_shape = image_shape

    # Train Model
    def train_model(self):
        x_train, x_test, y_train, y_test = self.split_data()
        print("Getting model...")
        model = self.get_model()

        print("Compiling the model...")
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(learning_rate=0.001),
                      metrics=['accuracy', OneHotMeanIoU(num_classes=self.n_classes)])

        callbacks = [ModelCheckpoint("UNet-Trained-Model-best.h5", save_best_only=True)]

        print("Training the model...")
        history = model.fit(x_train, y_train,
                            validation_data=(x_test, y_test),
                            batch_size=70,
                            epochs=35,
                            callbacks=callbacks)

    # Get UNet model from segmentation_model library
    def get_model(self):
        input_shape = self.image_shape
        n_classes = self.n_classes
        model = sm.Unet(backbone_name='resnet34', input_shape=input_shape, classes=n_classes, activation="softmax")
        return model

    # Split data
    def split_data(self):
        print('Splitting the dataset...')
        X_train, X_test, y_train, y_test = train_test_split(self.data.image_dataset, self.data.mask_dataset,
                                                            test_size=0.2,
                                                            random_state=42)

        train_masks_cat = to_categorical(y_train, num_classes=self.n_classes)
        y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], self.n_classes))

        test_masks_cat = to_categorical(y_test, num_classes=self.n_classes)
        y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], self.n_classes))

        return X_train, X_test, y_train_cat, y_test_cat


if __name__ == '__main__':
    images_path = "ade20k-4379-outdoor-images/images/"
    masks_path = "ade20k-4379-outdoor-images/masks/"

    n_classes = 18
    image_shape = (256, 256, 3)

    data = Data(images_path, masks_path)
    data.load_datasets()

    model_training = ModelTraining(data=data, n_classes=n_classes, image_shape=image_shape)

    model_training.train_model()
