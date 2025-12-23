import jax
import jax.numpy as jnp
import numpy as np
from collections import Counter



class KNeighborsClassifier:
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
