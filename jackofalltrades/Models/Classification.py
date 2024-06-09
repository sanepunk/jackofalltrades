from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, InputLayer
import tensorflow as tf
import numpy as np

class ImageClassification:
    def __init__(self, input_shape: tuple = (28, 28, 1), num_classes: int = 8, label_type: str = 'categorical',
                 normalizer: bool = True, metrics: list = None):
        """
        Initializes the ImageClassification instance.

        Args:
            input_shape (tuple): Shape of the input images. This is the size of the image that the model will accept.
                The default value is (28, 28, 1), which corresponds to a 28x28 grayscale image.

            num_classes (int): Number of output classes. This is the number of categories the model will predict.

            label_type (str): Type of label encoding. This is the format in which the labels are provided.
                - 'categorical': Labels are integers representing the class index.
                - 'onehotencoded': Labels are one-hot encoded arrays.

            normalizer (bool): Whether to normalize the input data. If True, the input data will be scaled to a range
                                of 0-1.

            metrics (list): List of metrics to be evaluated by the model. If not provided, accuracy will be used by
                            default.
        """
        try:
            self.input_shape = input_shape
            self.num_classes = num_classes
            self.label_type = label_type
            self.normalizer = normalizer
            self.metrics = metrics if metrics is not None else ['accuracy']
            self.cache_model = self.model()
        except Exception as e:
            raise Exception(f"Error initializing the ImageClassification instance: {e}")

    def model(self):
        # Initialize a Sequential model
        try:
            model = Sequential()
            # Add a 2D convolution layer with 32 output filters, a 3x3 kernel, and 'relu' activation function
            # The input shape is the shape of the input images
            model.add(InputLayer(shape=self.input_shape))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            # Add a Batch Normalization layer to normalize the activations of the previous layer
            model.add(BatchNormalization())
            # Add a Max Pooling layer with a 2x2 pool size to downsample the input
            model.add(MaxPool2D(pool_size=(2, 2)))

            # Repeat the same pattern of layers with 64 output filters in the Conv2D layer
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))

            # Repeat the same pattern of layers with 128 output filters in the Conv2D layer
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(BatchNormalization())
            model.add(MaxPool2D(pool_size=(2, 2)))

            # Add a Flatten layer to convert the 3D outputs to 1D vector
            model.add(Flatten())
            # Add a Dense layer (fully connected layer) with 128 units and 'relu' activation function
            # L2 regularization is applied to the weights
            model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
            # Add a Dropout layer with a rate of 0.5 to prevent overfitting
            model.add(Dropout(0.5))
            # Add another Dense layer with 64 units and 'relu' activation function
            model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
            model.add(Dropout(0.5))

            # If there are only 2 classes, add a Dense layer with 1 unit and 'sigmoid' activation function
            # Compile the model with 'adam' optimizer, 'binary_crossentropy' loss function, and the specified metrics
            if self.num_classes == 2:
                model.add(Dense(1, activation='sigmoid'))
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=self.metrics)

            # If there are more than 2 classes, add a Dense layer with units equal to the number of classes and
            # 'softmax' activation function
            # Compile the model with 'adam' optimizer, the appropriate loss function based on the label type, and
            # the specified metrics
            elif self.num_classes > 2:
                model.add(Dense(self.num_classes, activation='softmax'))
                if self.label_type.strip().lower() == 'categorical':
                    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=self.metrics)
                elif self.label_type.strip().lower() in ['onehotencoded', 'one-hot-encoded', 'one_hot_encoded']:
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=self.metrics)
            # Return the constructed model
            return model
        except Exception as e:
            raise Exception(f"Error creating the model: {e}")

    def fit(self, X_train, y_train, epochs: int = 10, batch_size: int = 32, validation_data=None,
            callbacks=None, verbose: int = 1, device: str = 'cpu', mirror_strategy: bool = False):
        """
        This method fits the model to the training data.

        Args:
            X_train: The training data.

            y_train: The labels for the training data.

            epochs (int): The number of times the learning algorithm will work through the entire training dataset.

            batch_size (int): The number of training examples utilized in one iteration.

            validation_data: The data on which to evaluate the loss and any model metrics at the end of each epoch.

            callbacks: List of callbacks to apply during training.

            verbose (int): Verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch.

            device (str): The device to run the training on. 'cpu' or 'gpu'.

            mirror_strategy (bool): If True, use MirroredStrategy for distributed training.

        Returns:
            None
        """
        # If normalizer is True, normalize the training data to a range of 0-1
        try:
            X_train = np.array(X_train, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1],
                                                                  self.input_shape[2])
            if self.normalizer:
                X_train = X_train.astype('float32') / 255.0

            # If cache_model is not None, use the cached model. Otherwise, build a new model
            model = self.cache_model if self.cache_model is not None else self.model()

            # If validation_data is provided, split it into validation inputs and labels
            if validation_data is not None:
                X_val, y_val = validation_data
                X_val = np.array(X_val, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
                # If normalizer is True, normalize the validation data to a range of 0-1
                if self.normalizer:
                    X_val = X_val.astype('float32') / 255.0
                # If mirror_strategy is True and device is 'gpu', use MirroredStrategy for distributed training
                if mirror_strategy and device == 'gpu':
                    mirrored_strategy = tf.distribute.MirroredStrategy()
                    with mirrored_strategy.scope():
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                                  callbacks=callbacks, verbose=verbose)
                # If device is 'gpu', run the training on GPU
                elif device == 'gpu':
                    with tf.device('/device:GPU:0'):
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                                  callbacks=callbacks, verbose=verbose)
                # If device is not 'gpu', run the training on CPU
                else:
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                              callbacks=callbacks, verbose=verbose)

            # If validation_data is not provided, just fit the model to the training data
            else:
                # If mirror_strategy is True and device is 'gpu', use MirroredStrategy for distributed training
                if mirror_strategy and device == 'gpu':
                    mirrored_strategy = tf.distribute.MirroredStrategy()
                    with mirrored_strategy.scope():
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
                # If device is 'gpu', run the training on GPU
                elif device == 'gpu':
                    with tf.device('/device:GPU:0'):
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
                # If device is not 'gpu', run the training on CPU
                else:
                    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
            # Cache the trained model
            self.cache_model = model
        except Exception as e:
            raise Exception(f"Error during training: {e}")

    def save(self, path):
        """
        This method saves the trained model to the specified path.

        Args:
            path (str): The path where the trained model will be saved. This can be a local path or a remote path.

        Returns:
            None
        """

        # Check if the cached model exists. If not, raise an exception.
        try:
            if self.cache_model is None:
                raise Exception("Model not trained yet.")

            # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
            model = self.cache_model

            # Use the save method of the model to save the model to the specified path.
            # The model will be saved in the TensorFlow SavedModel format by default.
            # This format includes both the model architecture and the trained weights.
            model.save(path)
        except Exception as e:
            raise Exception(f"Error saving the model: {e}")

    def predict(self, X):
        """
        This method makes predictions using the trained model.

        Args:
            X: The input data for which predictions will be made.

        Returns:
            predictions: The predicted values.
        """

        # Check if the cached model exists. If not, raise an exception.
        try:
            X = np.array(X, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            if self.cache_model is None:
                raise Exception("Model not trained yet.")

            # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
            model = self.cache_model

            # If normalizer is True, normalize the input data to a range of 0-1
            if self.normalizer:
                X = X.astype('float32') / 255.0

            # Use the predict method of the model to make predictions on the input data.
            predictions = model.predict(X)

            return predictions
        except Exception as e:
            raise Exception(f"Error making predictions: {e}")

    def evaluate(self, X_test, y_test):
        """
        This method evaluates the model on the test data.

        Args:
            X_test: The input test data.
            y_test: The labels for the test data.

        Returns:
            evaluation: The evaluation results.
        """
        try:
            X_test = np.array(X_test, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1],
                                                                self.input_shape[2])
            # Check if the cached model exists. If not, raise an exception.
            if self.cache_model is None:
                raise Exception("Model not trained yet.")

            # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
            model = self.cache_model

            # If normalizer is True, normalize the test data to a range of 0-1
            if self.normalizer:
                X_test = X_test.astype('float32') / 255.0

            # Use the evaluate method of the model to evaluate the model on the test data.
            evaluation = model.evaluate(X_test, y_test)

            return evaluation
        except Exception as e:
            raise Exception(f"Error evaluating the model: {e}")

    def summary(self):
        """
        This method prints a summary of the model architecture.

        Returns:
            None
        """

        try:
            # Check if the cached model exists. If not, raise an exception.
            if self.cache_model is None:
                raise Exception("Model not trained yet.")

            # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
            model = self.cache_model

            # Use the summary method of the model to print a summary of the model architecture.
            model.summary()
        except Exception as e:
            raise Exception(f"Error printing the model summary: {e}")

    def load(self, path):
        """
        This method loads a trained model from the specified path.

        Args:
            path (str): The path from which the trained model will be loaded. This can be a local path or a remote path.

        Returns:
            None
        """

        try:
            # Use the load_model method of the keras.models module to load the model from the specified path.
            # The loaded model will be stored in the cache_model attribute for future use.
            self.cache_model = tf.keras.models.load_model(path)
        except Exception as e:
            raise Exception(f"Error loading the model: {e}")

# Usage Example
# from jackofalltrades.Models.Classification import ImageClassification
# import numpy as np
# from sklearn.model_selection import train_test_split
# from jackofalltrades.datasets import get_data
#
# # Load the dataset
# ldset = get_data()
# X, y = ldset.get_mnist()
#


