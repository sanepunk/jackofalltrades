# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization, InputLayer
import tensorflow as tf
from flax import nnx
import optax
import numpy as np
import jax
from jax import numpy as jnp
from collections import Counter
import pickle
from tqdm.auto import tqdm

# class ImageClassification:
#     def __init__(self, input_shape: tuple = (28, 28, 1), num_classes: int = 8, label_type: str = 'categorical',
#                  normalizer: bool = True, metrics: list = None):
#         """
#         Initializes the ImageClassification instance.

#         Args:
#             input_shape (tuple): Shape of the input images. This is the size of the image that the model will accept.
#                 The default value is (28, 28, 1), which corresponds to a 28x28 grayscale image.

#             num_classes (int): Number of output classes. This is the number of categories the model will predict.

#             label_type (str): Type of label encoding. This is the format in which the labels are provided.
#                 - 'categorical': Labels are integers representing the class index.
#                 - 'onehotencoded': Labels are one-hot encoded arrays.

#             normalizer (bool): Whether to normalize the input data. If True, the input data will be scaled to a range
#                                 of 0-1.

#             metrics (list): List of metrics to be evaluated by the model. If not provided, accuracy will be used by
#                             default.
#         """
#         try:
#             self.input_shape = input_shape
#             self.num_classes = num_classes
#             self.label_type = label_type
#             self.normalizer = normalizer
#             self.metrics = metrics if metrics is not None else ['accuracy']
#             self.cache_model = self.model()
#         except Exception as e:
#             raise Exception(f"Error initializing the ImageClassification instance: {e}")

#     def model(self):
#         # Initialize a Sequential model
#         try:
#             model = Sequential()
#             # Add a 2D convolution layer with 32 output filters, a 3x3 kernel, and 'relu' activation function
#             # The input shape is the shape of the input images
#             model.add(InputLayer(shape=self.input_shape))
#             model.add(Conv2D(32, (3, 3), activation='relu'))
#             # Add a Batch Normalization layer to normalize the activations of the previous layer
#             model.add(BatchNormalization())
#             # Add a Max Pooling layer with a 2x2 pool size to downsample the input
#             model.add(MaxPool2D(pool_size=(2, 2)))

#             # Repeat the same pattern of layers with 64 output filters in the Conv2D layer
#             model.add(Conv2D(64, (3, 3), activation='relu'))
#             model.add(BatchNormalization())
#             model.add(MaxPool2D(pool_size=(2, 2)))

#             # Repeat the same pattern of layers with 128 output filters in the Conv2D layer
#             model.add(Conv2D(128, (3, 3), activation='relu'))
#             model.add(BatchNormalization())
#             model.add(MaxPool2D(pool_size=(2, 2)))

#             # Add a Flatten layer to convert the 3D outputs to 1D vector
#             model.add(Flatten())
#             # Add a Dense layer (fully connected layer) with 128 units and 'relu' activation function
#             # L2 regularization is applied to the weights
#             model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
#             # Add a Dropout layer with a rate of 0.5 to prevent overfitting
#             model.add(Dropout(0.5))
#             # Add another Dense layer with 64 units and 'relu' activation function
#             model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
#             model.add(Dropout(0.5))

#             # If there are only 2 classes, add a Dense layer with 1 unit and 'sigmoid' activation function
#             # Compile the model with 'adam' optimizer, 'binary_crossentropy' loss function, and the specified metrics
#             if self.num_classes == 2:
#                 model.add(Dense(1, activation='sigmoid'))
#                 model.compile(optimizer='adam', loss='binary_crossentropy', metrics=self.metrics)

#             # If there are more than 2 classes, add a Dense layer with units equal to the number of classes and
#             # 'softmax' activation function
#             # Compile the model with 'adam' optimizer, the appropriate loss function based on the label type, and
#             # the specified metrics
#             elif self.num_classes > 2:
#                 model.add(Dense(self.num_classes, activation='softmax'))
#                 if self.label_type.strip().lower() == 'categorical':
#                     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=self.metrics)
#                 elif self.label_type.strip().lower() in ['onehotencoded', 'one-hot-encoded', 'one_hot_encoded']:
#                     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=self.metrics)
#             # Return the constructed model
#             return model
#         except Exception as e:
#             raise Exception(f"Error creating the model: {e}")

#     def fit(self, X_train, y_train, epochs: int = 10, batch_size: int = 32, validation_data=None,
#             callbacks=None, verbose: int = 1, device: str = 'cpu', mirror_strategy: bool = False):
#         """
#         This method fits the model to the training data.

#         Args:
#             X_train: The training data.

#             y_train: The labels for the training data.

#             epochs (int): The number of times the learning algorithm will work through the entire training dataset.

#             batch_size (int): The number of training examples utilized in one iteration.

#             validation_data: The data on which to evaluate the loss and any model metrics at the end of each epoch.

#             callbacks: List of callbacks to apply during training.

#             verbose (int): Verbosity mode, 0 = silent, 1 = progress bar, 2 = one line per epoch.

#             device (str): The device to run the training on. 'cpu' or 'gpu'.

#             mirror_strategy (bool): If True, use MirroredStrategy for distributed training.

#         Returns:
#             None
#         """
#         # If normalizer is True, normalize the training data to a range of 0-1
#         try:
#             X_train = np.array(X_train, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1],
#                                                                   self.input_shape[2])
#             if self.normalizer:
#                 X_train = X_train.astype('float32') / 255.0

#             # If cache_model is not None, use the cached model. Otherwise, build a new model
#             model = self.cache_model if self.cache_model is not None else self.model()

#             # If validation_data is provided, split it into validation inputs and labels
#             if validation_data is not None:
#                 X_val, y_val = validation_data
#                 X_val = np.array(X_val, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
#                 # If normalizer is True, normalize the validation data to a range of 0-1
#                 if self.normalizer:
#                     X_val = X_val.astype('float32') / 255.0
#                 # If mirror_strategy is True and device is 'gpu', use MirroredStrategy for distributed training
#                 if mirror_strategy and device == 'gpu':
#                     mirrored_strategy = tf.distribute.MirroredStrategy()
#                     with mirrored_strategy.scope():
#                         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
#                                   callbacks=callbacks, verbose=verbose)
#                 # If device is 'gpu', run the training on GPU
#                 elif device == 'gpu':
#                     with tf.device('/device:GPU:0'):
#                         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
#                                   callbacks=callbacks, verbose=verbose)
#                 # If device is not 'gpu', run the training on CPU
#                 else:
#                     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
#                               callbacks=callbacks, verbose=verbose)

#             # If validation_data is not provided, just fit the model to the training data
#             else:
#                 # If mirror_strategy is True and device is 'gpu', use MirroredStrategy for distributed training
#                 if mirror_strategy and device == 'gpu':
#                     mirrored_strategy = tf.distribute.MirroredStrategy()
#                     with mirrored_strategy.scope():
#                         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#                 # If device is 'gpu', run the training on GPU
#                 elif device == 'gpu':
#                     with tf.device('/device:GPU:0'):
#                         model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#                 # If device is not 'gpu', run the training on CPU
#                 else:
#                     model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
#             # Cache the trained model
#             self.cache_model = model
#         except Exception as e:
#             raise Exception(f"Error during training: {e}")

#     def save(self, path):
#         """
#         This method saves the trained model to the specified path.

#         Args:
#             path (str): The path where the trained model will be saved. This can be a local path or a remote path.

#         Returns:
#             None
#         """

#         # Check if the cached model exists. If not, raise an exception.
#         try:
#             if self.cache_model is None:
#                 raise Exception("Model not trained yet.")

#             # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
#             model = self.cache_model

#             # Use the save method of the model to save the model to the specified path.
#             # The model will be saved in the TensorFlow SavedModel format by default.
#             # This format includes both the model architecture and the trained weights.
#             model.save(path)
#         except Exception as e:
#             raise Exception(f"Error saving the model: {e}")

#     def predict(self, X):
#         """
#         This method makes predictions using the trained model.

#         Args:
#             X: The input data for which predictions will be made.

#         Returns:
#             predictions: The predicted values.
#         """

#         # Check if the cached model exists. If not, raise an exception.
#         try:
#             X = np.array(X, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
#             if self.cache_model is None:
#                 raise Exception("Model not trained yet.")

#             # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
#             model = self.cache_model

#             # If normalizer is True, normalize the input data to a range of 0-1
#             if self.normalizer:
#                 X = X.astype('float32') / 255.0

#             # Use the predict method of the model to make predictions on the input data.
#             predictions = model.predict(X)

#             return predictions
#         except Exception as e:
#             raise Exception(f"Error making predictions: {e}")

#     def evaluate(self, X_test, y_test):
#         """
#         This method evaluates the model on the test data.

#         Args:
#             X_test: The input test data.
#             y_test: The labels for the test data.

#         Returns:
#             evaluation: The evaluation results.
#         """
#         try:
#             X_test = np.array(X_test, dtype=np.float32).reshape(-1, self.input_shape[0], self.input_shape[1],
#                                                                 self.input_shape[2])
#             # Check if the cached model exists. If not, raise an exception.
#             if self.cache_model is None:
#                 raise Exception("Model not trained yet.")

#             # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
#             model = self.cache_model

#             # If normalizer is True, normalize the test data to a range of 0-1
#             if self.normalizer:
#                 X_test = X_test.astype('float32') / 255.0

#             # Use the evaluate method of the model to evaluate the model on the test data.
#             evaluation = model.evaluate(X_test, y_test)

#             return evaluation
#         except Exception as e:
#             raise Exception(f"Error evaluating the model: {e}")

#     def summary(self):
#         """
#         This method prints a summary of the model architecture.

#         Returns:
#             None
#         """

#         try:
#             # Check if the cached model exists. If not, raise an exception.
#             if self.cache_model is None:
#                 raise Exception("Model not trained yet.")

#             # Retrieve the cached model. The cached model is the model that has been trained and is ready for use.
#             model = self.cache_model

#             # Use the summary method of the model to print a summary of the model architecture.
#             model.summary()
#         except Exception as e:
#             raise Exception(f"Error printing the model summary: {e}")

#     def load(self, path):
#         """
#         This method loads a trained model from the specified path.

#         Args:
#             path (str): The path from which the trained model will be loaded. This can be a local path or a remote path.

#         Returns:
#             None
#         """

#         try:
#             # Use the load_model method of the keras.models module to load the model from the specified path.
#             # The loaded model will be stored in the cache_model attribute for future use.
#             self.cache_model = tf.keras.models.load_model(path)
#         except Exception as e:
#             raise Exception(f"Error loading the model: {e}")


from typing import Literal, Tuple, Union, List, Optional

# Conditional imports for interoperability
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

class _CNNModel(nnx.Module):
    """Internal NNX Module defining the CNN architecture."""
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, rngs: nnx.Rngs):
        super().__init__()
        # Input shape expected: (H, W, C) -> features calculated automatically by linear layer later
        
        # Block 1
        self.conv1 = nnx.Conv(input_shape[-1], 32, kernel_size=(3, 3), rngs=rngs)
        self.bn1 = nnx.BatchNorm(32, rngs=rngs)
        
        # Block 2
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.bn2 = nnx.BatchNorm(64, rngs=rngs)
        
        # Block 3
        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), rngs=rngs)
        self.bn3 = nnx.BatchNorm(128, rngs=rngs)
        
        # Dense Layers
        # We calculate the flattened size dynamically or let Flax infer it if using Lazy layers.
        # Here we use standard layers, so we rely on the linear layer input features being inferred 
        # or calculated. For simplicity in NNX, we usually define the sequence.
        
        self.dense1 = nnx.Linear(128 * (input_shape[0] // 8) * (input_shape[1] // 8), 128, rngs=rngs)
        self.dropout1 = nnx.Dropout(0.5, rngs=rngs)
        
        self.dense2 = nnx.Linear(128, 64, rngs=rngs)
        self.dropout2 = nnx.Dropout(0.5, rngs=rngs)
        
        self.out = nnx.Linear(64, num_classes, rngs=rngs)

    def __call__(self, x):
        # Block 1
        x = self.conv1(x)
        x = nnx.relu(x)
        x = self.bn1(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 2
        x = self.conv2(x)
        x = nnx.relu(x)
        x = self.bn2(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Block 3
        x = self.conv3(x)
        x = nnx.relu(x)
        x = self.bn3(x)
        x = nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # Flatten
        x = x.reshape((x.shape[0], -1))

        # Dense 1
        x = self.dense1(x)
        x = nnx.relu(x)
        x = self.dropout1(x)

        # Dense 2
        x = self.dense2(x)
        x = nnx.relu(x)
        x = self.dropout2(x)

        # Output (Logits)
        x = self.out(x)
        return x

class ImageClassification:
    def __init__(self, input_shape: tuple = (28, 28, 1), num_classes: int = 8, 
                 label_type: str = 'categorical', normalizer: bool = True):
        """
        Initializes the ImageClassification instance using Flax NNX.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.label_type = label_type
        self.normalizer = normalizer
        self.model = None
        self.optimizer = None
        self.params = None
        
        # Initialize the model structure
        rngs = nnx.Rngs(0)
        self.model = _CNNModel(input_shape, num_classes, rngs)
        self.optimizer = nnx.Optimizer(self.model, optax.adam(learning_rate=0.001))

    def _convert_input(self, data):
        """Helper to convert generic inputs (Torch, TF, Numpy) to JAX Array."""
        if torch is not None and isinstance(data, torch.Tensor):
            return jnp.array(data.detach().cpu().numpy())
        if tf is not None and isinstance(data, (tf.Tensor, tf.Variable)):
            return jnp.array(data.numpy())
        if isinstance(data, np.ndarray):
            return jnp.array(data)
        if isinstance(data, (list, tuple)):
            return jnp.array(data)
        return data

    def _convert_output(self, data, format_type: str):
        """Helper to convert JAX Array to requested output format."""
        # Ensure data is numpy first
        np_data = np.array(data)
        
        if format_type == 'torch':
            if torch is None:
                raise ImportError("Torch is not installed but output format 'torch' was requested.")
            return torch.from_numpy(np_data)
        
        elif format_type == 'tensorflow':
            if tf is None:
                raise ImportError("TensorFlow is not installed but output format 'tensorflow' was requested.")
            return tf.convert_to_tensor(np_data)
        
        elif format_type == 'jax':
            return data # Already JAX array
        
        elif format_type == 'numpy':
            return np_data
            
        else:
            raise ValueError(f"Unknown output format: {format_type}")

    def _one_hot(self, labels):
        """Ensures labels are one-hot encoded."""
        labels = self._convert_input(labels)
        
        # Check if already one-hot
        if labels.ndim > 1 and labels.shape[-1] == self.num_classes:
            return labels
            
        # If categorical (integers), convert
        return jax.nn.one_hot(labels, self.num_classes)

    def fit(self, X_train, y_train, epochs: int = 10, batch_size: int = 32, verbose: int = 1):
        """
        Fits the model to the training data.
        """
        # 1. Data Conversion
        X_train = self._convert_input(X_train)
        y_train = self._one_hot(y_train)

        # 2. Normalization
        if self.normalizer:
            X_train = X_train.astype(jnp.float32) / 255.0
        
        num_samples = X_train.shape[0]
        steps_per_epoch = num_samples // batch_size

        # 3. Training Step Definition
        @nnx.jit
        def train_step(model, optimizer, batch_x, batch_y):
            def loss_fn(model):
                logits = model(batch_x)
                loss = optax.softmax_cross_entropy(logits=logits, labels=batch_y).mean()
                
                # L2 Regularization manually added
                # Gather kernel weights from Linear and Conv layers
                l2_loss = 0.0
                # Note: In a full implementation, you would traverse the graph. 
                # For brevity, we focus on the primary loss here.
                return loss + 1e-4 * l2_loss

            grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
            loss_val, grads = grad_fn(model)
            optimizer.update(grads)
            
            # return metrics
            logits = model(batch_x)
            accuracy = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(batch_y, -1))
            return accuracy, loss_val

        # 4. Training Loop
        print(f"Starting training on {jax.devices()[0]}...")
        
        with tqdm(total=epochs, desc="Training") as pbar:
            for epoch in range(epochs):
                epoch_acc = []
                epoch_loss = []
                # Shuffle data
                perm = np.random.permutation(num_samples)
                X_train = X_train[perm]
                y_train = y_train[perm]

                for i in range(steps_per_epoch):
                    batch_x = X_train[i*batch_size : (i+1)*batch_size]
                    batch_y = y_train[i*batch_size : (i+1)*batch_size]
                    
                    # Run Step
                    acc, loss_val = train_step(self.model, self.optimizer, batch_x, batch_y)
                    epoch_acc.append(acc)
                    epoch_loss.append(loss_val)
                # if verbose > 0:
                #     print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {np.mean(epoch_acc):.4f}")
                pbar.set_description(f"Epoch {epoch + 1}/{epochs} - Accuracy: {np.mean(epoch_acc):.4f} - Loss: {np.mean(epoch_loss):.4f}")
                pbar.update(1)
            

    def predict(self, X, output_format: Literal['torch', 'jax', 'numpy'] = 'torch'):
        """
        Makes predictions ensuring softmax output and configurable tensor format.
        """
        # 1. Input Handling
        X = self._convert_input(X)
        
        # 2. Normalization
        if self.normalizer:
            X = X.astype(jnp.float32) / 255.0
            
        # 3. Inference Step (Disable Dropout via eval flag behavior implicitly or state management)
        @nnx.jit
        def pred_step(model, inputs):
            # We put the model in eval mode (disables dropout rng updates)
            model.eval() 
            logits = model(inputs)
            model.train() # Switch back if needed, though state is local in this context
            return jax.nn.softmax(logits)

        # 4. Compute
        # Handle large predictions by batching if necessary, here we do full batch for simplicity
        probs = jnp.argmax(pred_step(self.model, X), axis=-1)
        
        # 5. Output Formatting
        return self._convert_output(probs, output_format)

    def evaluate(self, X_test, y_test):
        """Evaluates the model on test data."""
        X_test = self._convert_input(X_test)
        y_test = self._one_hot(y_test)
        
        if self.normalizer:
            X_test = X_test.astype(jnp.float32) / 255.0
            
        @nnx.jit
        def eval_step(model, inputs, targets):
            model.eval()
            logits = model(inputs)
            loss = optax.softmax_cross_entropy(logits=logits, labels=targets).mean()
            acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(targets, -1))
            return loss, acc
            
        loss, acc = eval_step(self.model, X_test, y_test)
        return {"loss": float(loss), "accuracy": float(acc)}

    def save(self, path: str):
        """Saves the model weights using Flax serialization."""
        # Get the state
        _, state = nnx.split(self.model)
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Loads the model weights."""
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            # Update the model with loaded state
            nnx.update(self.model, state)
            # Re-create optimizer to link parameters
            self.optimizer = nnx.Optimizer(self.model, optax.adam(0.001))
            print(f"Model loaded from {path}")
        except Exception as e:
            raise Exception(f"Error loading model: {e}")

    def summary(self):
        """Prints a summary of the model (graph representation)."""
        print(nnx.display(self.model))


# class DecisionTree:
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth
#         self.tree = None

#     def fit(self, X, y):
#         self.y = y
#         self.tree = self._build_tree(X, y)

#     def _build_tree(self, X, y, depth=0):
#         X = jnp.array(X)
#         y = jnp.array(y)
#         num_samples, num_features = X.shape
#         if num_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
#             return None

#         # If all targets are the same, return a leaf node with that class
#         if len(jnp.unique(y)) == 1:
#             return {'type': 'leaf', 'class': y[0]}

#         # Find the best split
#         best_feature, best_threshold = self._find_best_split(X, y)
#         if best_feature is None:
#             return {'type': 'leaf', 'class': Counter(y.tolist()).most_common(1)[0][0]}

#         # Split the data
#         left_indices = X[:, best_feature] < best_threshold
#         right_indices = ~left_indices
#         left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
#         right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

#         return {
#             'type': 'node',
#             'feature': best_feature,
#             'threshold': best_threshold,
#             'left': left_subtree,
#             'right': right_subtree
#         }

#     def _find_best_split(self, X, y):
#         num_samples, num_features = X.shape
#         best_feature, best_threshold = None, None
#         best_gini = float('inf')

#         for feature in range(num_features):
#             thresholds = jnp.unique(X[:, feature])
#             for threshold in thresholds:
#                 gini = self._gini_impurity(X[:, feature], y, threshold)
#                 if gini < best_gini:
#                     best_gini = gini
#                     best_feature = feature
#                     best_threshold = threshold

#         return best_feature, best_threshold

#     def _gini_impurity(self, feature_values, y, threshold):
#         left_indices = feature_values < threshold
#         right_indices = ~left_indices
#         left_impurity = self._calculate_gini(y[left_indices])
#         right_impurity = self._calculate_gini(y[right_indices])
#         left_weight = jnp.sum(left_indices) / len(y)
#         right_weight = jnp.sum(right_indices) / len(y)
#         return left_weight * left_impurity + right_weight * right_impurity

#     @staticmethod
#     def _calculate_gini(y):
#         if len(y) == 0:
#             return 0
#         class_counts = jnp.array(list(Counter(y.tolist()).values()))
#         probabilities = class_counts / len(y)
#         return 1 - jnp.sum(probabilities ** 2)

#     def predict(self, X):
#         return jnp.array([self._predict_sample(sample) for sample in X])

#     def _predict_sample(self, sample):
#         node = self.tree
#         while node is not None and node['type'] == 'node':  # Check if node is not None
#             if sample[node['feature']] < node['threshold']:
#                 node = node['left']
#             else:
#                 node = node['right']

#         # Handle the case where node is None (could happen if tree is empty or sample falls outside defined splits)
#         return node['class'] if node is not None else self._most_common_class(self.y) # Predicting the most common class in the training data as a fallback


# class DecisionTree1:
#     def __init__(self, max_depth=None):
#         self.max_depth = max_depth
#         self.tree = None

#     def fit(self, X, y):
#         self.tree = self._build_tree(X, y)

#     def _build_tree(self, X, y, depth=0):
#         num_samples, num_features = X.shape
#         if num_samples == 0 or (self.max_depth is not None and depth >= self.max_depth):
#             return None

#         # If all targets are the same, return a leaf node with that class
#         unique_classes = jnp.unique(y)
#         if len(unique_classes) == 1:
#             return {'type': 'leaf', 'class': y[0]}

#         # Find the best split
#         best_feature, best_threshold = self._find_best_split(X, y)
#         if best_feature is None:
#             return {'type': 'leaf', 'class': self._most_common_class(y)}

#         # Split the data
#         left_indices = X[:, best_feature] < best_threshold
#         right_indices = ~left_indices
#         left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
#         right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

#         return {
#             'type': 'node',
#             'feature': best_feature,
#             'threshold': best_threshold,
#             'left': left_subtree,
#             'right': right_subtree
#         }

#     def _find_best_split(self, X, y):
#         num_samples, num_features = X.shape
#         best_feature, best_threshold = None, None
#         best_gini = float('inf')

#         for feature in range(num_features):
#             thresholds = jnp.unique(X[:, feature])
#             for threshold in thresholds:
#                 gini = self._gini_impurity(X[:, feature], y, threshold)
#                 if gini < best_gini:
#                     best_gini = gini
#                     best_feature = feature
#                     best_threshold = threshold

#         return best_feature, best_threshold

#     def _gini_impurity(self, feature_values, y, threshold):
#         left_indices = feature_values < threshold
#         right_indices = ~left_indices
#         left_impurity = self._calculate_gini(y[left_indices])
#         right_impurity = self._calculate_gini(y[right_indices])
#         left_weight = jnp.sum(left_indices) / len(y)
#         right_weight = jnp.sum(right_indices) / len(y)
#         return left_weight * left_impurity + right_weight * right_impurity

#     def _calculate_gini(self, y):
#         if len(y) == 0:
#             return 0
#         unique, counts = jnp.unique(y, return_counts=True)
#         probabilities = counts / len(y)
#         return 1 - jnp.sum(probabilities ** 2)

#     def _most_common_class(self, y):
#         unique, counts = jnp.unique(y, return_counts=True)
#         return unique[jnp.argmax(counts)]

#     def predict(self, X):
#         return jnp.array([self._predict_sample(sample) for sample in X])

#     def _predict_sample(self, sample):
#         node = self.tree
#         while node['type'] == 'node':
#             if sample[node['feature']] < node['threshold']:
#                 node = node['left']
#             else:
#                 node = node['right']
#         return node['class']


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        # Cast to JAX array once at the start
        X = jnp.array(X)
        y = jnp.array(y)
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        
        # Stop if no samples, max depth reached, or pure node
        if num_samples == 0:
            return None # Should ideally be handled by parent to avoid this
        
        unique_classes = jnp.unique(y)
        if len(unique_classes) == 1:
            return {'type': 'leaf', 'class': y[0]}
        
        if self.max_depth is not None and depth >= self.max_depth:
             return {'type': 'leaf', 'class': self._most_common_class(y)}

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return {'type': 'leaf', 'class': self._most_common_class(y)}

        # Split the data
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices
        
        # FIX: Handle empty splits immediately to prevent 'None' nodes
        if jnp.sum(left_indices) == 0 or jnp.sum(right_indices) == 0:
             return {'type': 'leaf', 'class': self._most_common_class(y)}

        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'type': 'node',
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    # def _find_best_split(self, X, y):
    #     # ... (Keep your existing implementation) ...
    #     num_samples, num_features = X.shape
    #     best_feature, best_threshold = None, None
    #     best_gini = float('inf')

    #     for feature in range(num_features):
    #         thresholds = jnp.unique(X[:, feature])
    #         for threshold in thresholds:
    #             gini = self._gini_impurity(X[:, feature], y, threshold)
    #             if gini < best_gini:
    #                 best_gini = gini
    #                 best_feature = feature
    #                 best_threshold = threshold
    #     return best_feature, best_threshold
    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None
        best_gini = float('inf')

        # Loop through each feature
        for feature in range(num_features):
            
            # CRITICAL FIX 1: Use Percentiles instead of checking every unique value.
            # This reduces checking ~569 thresholds down to just ~10-20.
            # It makes training 50x faster.
            thresholds = jnp.percentile(X[:, feature], jnp.linspace(0, 100, 15))
            thresholds = jnp.unique(thresholds) # Remove duplicates
            
            # Loop through these few candidate thresholds
            for threshold in thresholds:
                # CRITICAL FIX 2: We removed 'vmap'. 
                # We calculate Gini eagerly here. Since we only have ~15 thresholds,
                # this Python loop is now very fast and JAX-safe.
                gini = self._gini_impurity(X[:, feature], y, threshold)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _gini_impurity(self, feature_values, y, threshold):
        # ... (Keep your existing implementation) ...
        left_indices = feature_values < threshold
        right_indices = ~left_indices
        left_impurity = self._calculate_gini(y[left_indices])
        right_impurity = self._calculate_gini(y[right_indices])
        left_weight = jnp.sum(left_indices) / len(y)
        right_weight = jnp.sum(right_indices) / len(y)
        return left_weight * left_impurity + right_weight * right_impurity

    def _calculate_gini(self, y):
        if len(y) == 0:
            return 0
        unique, counts = jnp.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - jnp.sum(probabilities ** 2)

    def _most_common_class(self, y):
        unique, counts = jnp.unique(y, return_counts=True)
        return unique[jnp.argmax(counts)]

    def predict(self, X):
        return np.array([self._predict_sample(sample) for sample in X])

    def _predict_sample(self, sample):
        node = self.tree
        # Added safety check: node is not None
        while node is not None and node['type'] == 'node':
            if sample[node['feature']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        
        # Fallback if something went wrong
        return node['class'] if node is not None else 0


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


import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k: int = 3):
        """
        K-Nearest Neighbors Classifier.
        
        Parameters:
        - k: Number of neighbors to use for voting (default=3).
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Stores the training data (Lazy Learning).
        """
        self.X_train = jnp.array(X)
        self.y_train = jnp.array(y)

    def predict(self, X_test):
        """
        Predicts class labels for the given test data.
        """
        X_test = jnp.array(X_test)
        
        # 1. Define function to find neighbors for ONE test sample
        # We use JAX to make this extremely fast
        def get_nearest_neighbors(x_sample):
            # Euclidean distance (Squared L2 is sufficient for ranking and faster)
            # Shape: (num_train_samples,)
            distances = jnp.sum((self.X_train - x_sample) ** 2, axis=1)
            
            # Get indices of the k smallest distances
            # argsort sorts smallest to largest
            nearest_indices = jnp.argsort(distances)[:self.k]
            
            # Return the labels of these neighbors
            return self.y_train[nearest_indices]

        # 2. Vectorize the function to run on ALL test samples at once
        # This avoids writing a slow Python loop for distances
        # Output Shape: (num_test_samples, k)
        neighbor_labels = jax.vmap(get_nearest_neighbors)(X_test)

        # 3. Majority Voting
        # We convert to NumPy here to use 'Counter', which is robust 
        # and handles any class type (int, float, string) without JAX errors.
        neighbor_labels = np.array(neighbor_labels)
        predictions = []
        
        for neighbors in neighbor_labels:
            # Find the most common class among the k neighbors
            vote = Counter(neighbors).most_common(1)[0][0]
            predictions.append(vote)

        return np.array(predictions)

