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
        Enhanced vectorized prediction with optimized distance calculations and voting.
        """
        X_test = jnp.array(X_test)
        
        # Optimized approach for large datasets: use chunked processing
        if X_test.shape[0] > 5000 or self.X_train.shape[0] > 10000:
            return self._predict_chunked(X_test)
        
        # 1. Vectorized distance computation using broadcasting
        # Shape: (n_test, n_train)
        distances = jnp.sum((X_test[:, None, :] - self.X_train[None, :, :]) ** 2, axis=2)
        
        # 2. Get k nearest neighbors for all test samples at once
        # Shape: (n_test, k)
        nearest_indices = jnp.argsort(distances, axis=1)[:, :self.k]
        
        # 3. Get neighbor labels vectorized
        # Shape: (n_test, k)
        neighbor_labels = self.y_train[nearest_indices]
        
        # 4. Optimized majority voting using JAX operations
        return self._vectorized_voting(neighbor_labels)

    def _predict_chunked(self, X_test, chunk_size=1000):
        """
        Memory-efficient prediction for large datasets using chunking.
        """
        predictions = []
        
        for i in range(0, X_test.shape[0], chunk_size):
            chunk = X_test[i:i + chunk_size]
            
            # Use the original vmap approach for chunks
            def get_nearest_neighbors(x_sample):
                distances = jnp.sum((self.X_train - x_sample) ** 2, axis=1)
                nearest_indices = jnp.argsort(distances)[:self.k]
                return self.y_train[nearest_indices]

            neighbor_labels = jax.vmap(get_nearest_neighbors)(chunk)
            chunk_predictions = self._vectorized_voting(neighbor_labels)
            predictions.append(chunk_predictions)
        
        return np.concatenate(predictions, axis=0)

    def _vectorized_voting(self, neighbor_labels):
        """
        Efficient vectorized majority voting using JAX operations.
        """
        neighbor_labels = np.array(neighbor_labels)
        
        # For integer labels, we can use a more efficient bincount approach
        try:
            if np.issubdtype(neighbor_labels.dtype, np.integer):
                return self._integer_voting(neighbor_labels)
        except:
            pass
        
        # Fallback to Counter for non-integer or mixed types
        predictions = []
        for neighbors in neighbor_labels:
            vote = Counter(neighbors).most_common(1)[0][0]
            predictions.append(vote)
        
        return np.array(predictions)

    def _integer_voting(self, neighbor_labels):
        """
        Optimized voting for integer labels using vectorized operations.
        """
        max_label = int(np.max(neighbor_labels)) + 1
        predictions = []
        
        for neighbors in neighbor_labels:
            counts = np.bincount(neighbors.astype(int), minlength=max_label)
            predictions.append(np.argmax(counts))
        
        return np.array(predictions)
