"""
Tests for GAN model - computationally efficient with minimal training.
"""

import unittest
import numpy as np
import torch
from jackofalltrades.Models import GAN


class TestGAN(unittest.TestCase):
    """Tests for GAN model."""

    def setUp(self):
        """Set up test fixtures with very small dataset."""
        np.random.seed(42)
        # Very small dataset for testing
        n_samples = 16  # Small batch
        self.X = np.random.randn(n_samples, 64, 64, 1).astype(np.float32)

    def test_initialization(self):
        """Test GAN initialization."""
        gan = GAN(
            noise_dim=10,  # Small noise dimension
            image_channels=1,
            feature_maps_g=16,  # Small feature maps
            feature_maps_d=16,
        )
        self.assertIsNotNone(gan.generator)
        self.assertIsNotNone(gan.discriminator)

    def test_generate_before_training(self):
        """Test image generation before training."""
        gan = GAN(noise_dim=10, image_channels=1, feature_maps_g=16, feature_maps_d=16)
        generated = gan.generate(num_images=4, output_format="numpy")

        self.assertEqual(len(generated), 4)
        self.assertEqual(generated.shape[1:], (64, 64, 1))

    def test_train_minimal(self):
        """Test training with minimal epochs."""
        gan = GAN(noise_dim=10, image_channels=1, feature_maps_g=16, feature_maps_d=16)
        # Train with very few epochs and small batch
        gan.train(
            self.X,
            epochs=1,  # Single epoch
            batch_size=8,  # Small batch
            verbose=0,
        )

        # Should complete without errors
        self.assertTrue(True)

    def test_generate_after_training(self):
        """Test image generation after training."""
        gan = GAN(noise_dim=10, image_channels=1, feature_maps_g=16, feature_maps_d=16)
        gan.train(self.X, epochs=1, batch_size=8, verbose=0)

        generated = gan.generate(num_images=4, output_format="numpy")
        self.assertEqual(len(generated), 4)
        self.assertEqual(generated.shape[1:], (64, 64, 1))

    def test_save_load(self):
        """Test GAN save and load functionality."""
        import os
        import tempfile

        gan1 = GAN(noise_dim=10, image_channels=1, feature_maps_g=16, feature_maps_d=16)
        gan1.train(self.X, epochs=1, batch_size=8, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            gan1.save(tmpdir)

            gan2 = GAN(
                noise_dim=10, image_channels=1, feature_maps_g=16, feature_maps_d=16
            )
            gan2.load(tmpdir)

            # Generate images from both models
            gen1 = gan1.generate(2, output_format="numpy")
            gen2 = gan2.generate(2, output_format="numpy")

            self.assertEqual(gen1.shape, gen2.shape)

    def test_different_output_formats(self):
        """Test different output formats."""
        gan = GAN(noise_dim=10, image_channels=1, feature_maps_g=16, feature_maps_d=16)

        # Test numpy format
        numpy_out = gan.generate(2, output_format="numpy")
        self.assertIsInstance(numpy_out, np.ndarray)

        # Test jax format
        jax_out = gan.generate(2, output_format="jax")
        import jax.numpy as jnp

        self.assertIsInstance(jax_out, jnp.ndarray)

        # Test torch format
        torch_out = gan.generate(2, output_format="torch")
        self.assertIsInstance(torch_out, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
