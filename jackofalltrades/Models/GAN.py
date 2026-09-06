# import torch
# import torch.nn as nn

# class Generator(nn.Module):
# 	def __init__(self, input_dim, output_channels, feature_maps=64):
# 		"""
# 		Initializes the Generator with user-defined input dimension and output channels.

# 		:param input_dim: Dimension of the input noise vector (e.g., 100).
# 		:param output_channels: Number of channels in the output image (e.g., 3 for RGB images).
# 		:param feature_maps: Base number of feature maps (default: 64).
# 		"""
# 		super(Generator, self).__init__()
# 		self.net = nn.Sequential(
# 			# Input: N x input_dim x 1 x 1
# 			nn.ConvTranspose2d(input_dim, feature_maps * 8, 4, 1, 0),  # Output: N x (feature_maps*8) x 4 x 4
# 			nn.BatchNorm2d(feature_maps * 8),
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # Output: N x (feature_maps*4) x 8 x 8
# 			nn.BatchNorm2d(feature_maps * 4),
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # Output: N x (feature_maps*2) x 16 x 16
# 			nn.BatchNorm2d(feature_maps * 2),
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),  # Output: N x feature_maps x 32 x 32
# 			nn.BatchNorm2d(feature_maps),
# 			nn.ReLU(True),
# 			nn.ConvTranspose2d(feature_maps, output_channels, 4, 2, 1),  # Output: N x output_channels x 64 x 64
# 			nn.Tanh()
# 		)

# 	def forward(self, x):
# 		return self.net(x)

# import torch.nn as nn

# class Discriminator(nn.Module):
# 	def __init__(self, input_channels, feature_dim):
# 		super(Discriminator, self).__init__()
# 		self.model = nn.Sequential(
# 			nn.Conv2d(input_channels, feature_dim, kernel_size=4, stride=2, padding=1),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
# 			nn.BatchNorm2d(feature_dim * 2),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
# 			nn.BatchNorm2d(feature_dim * 4),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
# 			nn.BatchNorm2d(feature_dim * 8),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Flatten(),  # Flatten the tensor before the linear layer
# 			nn.Linear(feature_dim * 8 * 4 * 4, 1),  # Adjust the input features based on the output size after convolutions
# 			nn.Sigmoid()  # Sigmoid activation for binary classification
# 		)

# 	def forward(self, x):
# 		return self.model(x)


# class GAN:
# 	def __init__(self, noise_dim, image_channels, feature_maps_g=64, feature_maps_d=64):
# 		"""
# 		Initializes the GAN with user-defined parameters for the generator and discriminator.

# 		:param noise_dim: Dimension of the input noise vector for the generator.
# 		:param image_channels: Number of channels in the generated and real images.
# 		:param feature_maps_g: Base number of feature maps for the generator (default: 64).
# 		:param feature_maps_d: Base number of feature maps for the discriminator (default: 64).
# 		"""
# 		self.generator = Generator(noise_dim, image_channels, feature_maps_g)
# 		self.discriminator = Discriminator(image_channels, feature_maps_d)
# 		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 		self.generator.to(self.device)
# 		self.discriminator.to(self.device)
# 		self.noise_dim = noise_dim

# 	def train(self, real_data_loader, epochs, learning_rate):
# 		"""
# 		Trains the GAN using the provided real data.

# 		:param real_data_loader: DataLoader for the real dataset.
# 		:param epochs: Number of training epochs.
# 		:param learning_rate: Learning rate for the optimizers.
# 		"""
# 		criterion = nn.BCELoss()
# 		self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# 		self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# 		for epoch in range(epochs):
# 			for i, real_images in enumerate(real_data_loader):
# 				real_images = real_images.to(self.device)
# 				batch_size = real_images.size(0)

# 				# Create labels
# 				real_labels = torch.ones(batch_size, 1).to(self.device)
# 				fake_labels = torch.zeros(batch_size, 1).to(self.device)

# 				# Train Discriminator
# 				self.discriminator.zero_grad()
# 				outputs = self.discriminator(real_images)
# 				d_loss_real = criterion(outputs, real_labels)
# 				d_loss_real.backward()

# 				noise = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
# 				fake_images = self.generator(noise)
# 				outputs = self.discriminator(fake_images.detach())
# 				d_loss_fake = criterion(outputs, fake_labels)
# 				d_loss_fake.backward()
# 				self.optimizer_d.step()

# 				# Train Generator
# 				self.generator.zero_grad()
# 				outputs = self.discriminator(fake_images)
# 				g_loss = criterion(outputs, real_labels)
# 				g_loss.backward()
# 				self.optimizer_g.step()

# 				if (i+1) % 100 == 0:
# 					print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(real_data_loader)}], '
# 						  f'Discriminator Loss: {d_loss_real.item() + d_loss_fake.item()}, '
# 						  f'Generator Loss: {g_loss.item()}')

# 	def generate(self, num_images):
# 		"""
# 		Generates images using the trained generator.

# 		:param num_images: Number of images to generate.
# 		:return: Generated images.
# 		"""
# 		self.generator.eval()
# 		with torch.no_grad():
# 			noise = torch.randn(num_images, self.noise_dim, 1, 1).to(self.device)
# 			generated_images = self.generator(noise)
# 		self.generator.train()
# 		return generated_images

# 	def save(self, path):
# 		"""
# 		Saves the generator and discriminator models.

# 		:param path: Path to save the models.
# 		"""
# 		torch.save(self.generator.state_dict(), path + '/generator.pth')
# 		torch.save(self.discriminator.state_dict(), path + '/discriminator.pth')
# 		torch.save(self.optimizer_g.state_dict(), path + '/optimizer_g.pth')
# 		torch.save(self.optimizer_d.state_dict(), path + '/optimizer_d.pth')

# 	def load(self, path):
# 		"""
# 		Loads the generator and discriminator models.

# 		:param path: Path to load the models.
# 		"""
# 		self.generator.load_state_dict(torch.load(path + '/generator.pth'))
# 		self.discriminator.load_state_dict(torch.load(path + '/discriminator.pth'))
# 		self.optimizer_g.load_state_dict(torch.load(path + '/optimizer_g.pth'))
# 		self.optimizer_d.load_state_dict(torch.load(path + '/optimizer_d.pth'))

import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from typing import Literal
from tqdm.auto import tqdm

# Conditional imports for interoperability
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


class _Generator(nnx.Module):
    def __init__(self, noise_dim, output_channels, feature_maps, rngs: nnx.Rngs):
        super().__init__()
        self.feature_maps = feature_maps

        self.fc = nnx.Linear(noise_dim, feature_maps * 8 * 4 * 4, rngs=rngs)
        self.bn0 = nnx.BatchNorm(feature_maps * 8, rngs=rngs)

        # padding='SAME' ensures correct 2x scaling with stride=2
        self.conv1 = nnx.ConvTranspose(
            feature_maps * 8,
            feature_maps * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn1 = nnx.BatchNorm(feature_maps * 4, rngs=rngs)

        self.conv2 = nnx.ConvTranspose(
            feature_maps * 4,
            feature_maps * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(feature_maps * 2, rngs=rngs)

        self.conv3 = nnx.ConvTranspose(
            feature_maps * 2,
            feature_maps,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn3 = nnx.BatchNorm(feature_maps, rngs=rngs)

        self.conv4 = nnx.ConvTranspose(
            feature_maps,
            output_channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.fc(x)
        x = x.reshape((x.shape[0], 4, 4, self.feature_maps * 8))
        x = self.bn0(x)
        x = nnx.relu(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nnx.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nnx.relu(x)
        x = self.conv4(x)
        x = nnx.tanh(x)
        return x


class _Discriminator(nnx.Module):
    def __init__(self, input_channels, feature_dim, rngs: nnx.Rngs):
        super().__init__()

        self.conv1 = nnx.Conv(
            input_channels,
            feature_dim,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )

        self.conv2 = nnx.Conv(
            feature_dim,
            feature_dim * 2,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn2 = nnx.BatchNorm(feature_dim * 2, rngs=rngs)

        self.conv3 = nnx.Conv(
            feature_dim * 2,
            feature_dim * 4,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn3 = nnx.BatchNorm(feature_dim * 4, rngs=rngs)

        self.conv4 = nnx.Conv(
            feature_dim * 4,
            feature_dim * 8,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="SAME",
            rngs=rngs,
        )
        self.bn4 = nnx.BatchNorm(feature_dim * 8, rngs=rngs)

        self.linear = nnx.Linear(feature_dim * 8 * 4 * 4, 1, rngs=rngs)

    def __call__(self, x):
        x = self.conv1(x)
        x = nnx.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = nnx.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = nnx.leaky_relu(x, negative_slope=0.2)
        x = self.conv4(x)
        x = self.bn4(x)
        x = nnx.leaky_relu(x, negative_slope=0.2)
        x = x.reshape((x.shape[0], -1))
        x = self.linear(x)
        x = nnx.sigmoid(x)
        return x


class GAN:
    def __init__(
        self,
        noise_dim: int,
        image_channels: int,
        feature_maps_g: int = 64,
        feature_maps_d: int = 64,
    ):
        self.noise_dim = noise_dim
        self.image_channels = image_channels
        self.feature_maps_g = feature_maps_g
        self.feature_maps_d = feature_maps_d

        rngs = nnx.Rngs(0)
        self.generator = _Generator(noise_dim, image_channels, feature_maps_g, rngs)
        self.discriminator = _Discriminator(image_channels, feature_maps_d, rngs)

        self.optimizer_g = nnx.Optimizer(
            self.generator, optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999)
        )
        self.optimizer_d = nnx.Optimizer(
            self.discriminator, optax.adam(learning_rate=0.0002, b1=0.5, b2=0.999)
        )

    def _convert_input(self, data):
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
        np_data = np.array(data)
        if format_type == "torch":
            if torch is None:
                raise ImportError("Torch is not installed.")
            return torch.from_numpy(np_data)
        elif format_type == "tensorflow":
            if tf is None:
                raise ImportError("TensorFlow is not installed.")
            return tf.convert_to_tensor(np_data)
        elif format_type == "jax":
            return data
        elif format_type == "numpy":
            return np_data
        else:
            raise ValueError(f"Unknown output format: {format_type}")

    def train(self, real_data, epochs: int, batch_size: int = 128, verbose: int = 1):
        X_train = self._convert_input(real_data)

        # Auto-Resize to 64x64 to match Generator output
        if X_train.shape[1] != 64 or X_train.shape[2] != 64:
            if verbose > 0:
                print(
                    f"Resizing input from {X_train.shape[1:3]} to (64, 64) to match Generator architecture..."
                )
            X_train = jax.image.resize(
                X_train,
                shape=(X_train.shape[0], 64, 64, X_train.shape[3]),
                method="bilinear",
            )

        # Normalize to [-1, 1]
        X_train = (X_train.astype(jnp.float32) - 127.5) / 127.5

        num_samples = X_train.shape[0]
        steps_per_epoch = num_samples // batch_size

        @nnx.jit
        def train_step_D(
            discriminator, generator, optimizer_d, real_batch, noise_batch
        ):
            # 1. PREVENT MUTATION: Generator must be in EVAL mode while training Discriminator
            generator.eval()
            fake_images = generator(noise_batch)

            def loss_fn_d(discriminator):
                # Discriminator stays in TRAIN mode here (default) to update its stats
                real_logits = discriminator(real_batch)
                d_loss_real = -jnp.mean(jnp.log(real_logits + 1e-8))

                fake_logits = discriminator(fake_images)
                d_loss_fake = -jnp.mean(jnp.log(1 - fake_logits + 1e-8))
                return d_loss_real + d_loss_fake

            grad_fn = nnx.value_and_grad(loss_fn_d)
            d_loss, grads = grad_fn(discriminator)
            optimizer_d.update(grads)
            return d_loss

        @nnx.jit
        def train_step_G(discriminator, generator, optimizer_g, noise_batch):
            # 2. PREVENT MUTATION: Discriminator must be in EVAL mode while training Generator
            # This ensures BN stats in Discriminator are NOT updated here.
            discriminator.eval()

            def loss_fn_g(generator):
                fake_images = generator(noise_batch)
                fake_output = discriminator(fake_images)
                # Generator loss: maximize log(D(G(z)))
                g_loss = -jnp.mean(jnp.log(fake_output + 1e-8))
                return g_loss

            grad_fn = nnx.value_and_grad(loss_fn_g)
            g_loss, grads = grad_fn(generator)
            optimizer_g.update(grads)
            return g_loss

        print(f"Starting GAN training on {jax.devices()[0]}...")
        with tqdm(total=epochs, desc="GAN Training", disable=verbose == 0) as pbar:
            for epoch in range(epochs):
                d_losses = []
                g_losses = []

                perm = np.random.permutation(num_samples)
                X_train = X_train[perm]

                for i in range(steps_per_epoch):
                    real_batch = X_train[i * batch_size : (i + 1) * batch_size]

                    # Train Discriminator
                    noise = jax.random.normal(
                        nnx.Rngs(epoch * steps_per_epoch + i).key(),
                        (batch_size, self.noise_dim),
                    )

                    # Reset D to train mode (it might have been set to eval by train_step_G previously)
                    self.discriminator.train()
                    d_loss = train_step_D(
                        self.discriminator,
                        self.generator,
                        self.optimizer_d,
                        real_batch,
                        noise,
                    )
                    d_losses.append(d_loss)

                    # Train Generator
                    noise_g = jax.random.normal(
                        nnx.Rngs((epoch * steps_per_epoch + i) + 1000).key(),
                        (batch_size, self.noise_dim),
                    )

                    # Reset G to train mode (it was set to eval by train_step_D)
                    self.generator.train()
                    g_loss = train_step_G(
                        self.discriminator, self.generator, self.optimizer_g, noise_g
                    )
                    g_losses.append(g_loss)

                pbar.set_description(
                    f"Epoch {epoch + 1}/{epochs} | D: {np.mean(d_losses):.4f} | G: {np.mean(g_losses):.4f}"
                )
                pbar.update(1)

    def generate(
        self, num_images: int, output_format: Literal["torch", "jax", "numpy"] = "torch"
    ):
        """Optimized image generation with support for large batches."""
        rng_key = nnx.Rngs(np.random.randint(0, 10000)).key()
        noise = jax.random.normal(rng_key, (num_images, self.noise_dim))

        # For very large batches, use chunked generation to avoid memory issues
        if num_images > 1000:
            return self._generate_chunked(noise, output_format)

        @nnx.jit
        def pred_step(model, z):
            model.eval()
            return model(z)

        generated_images = pred_step(self.generator, noise)
        return self._convert_output(generated_images, output_format)

    def _generate_chunked(self, noise, output_format: str, chunk_size=500):
        """Memory-efficient generation for large batches using chunking."""
        generated_chunks = []

        @nnx.jit
        def pred_step(model, z_chunk):
            model.eval()
            return model(z_chunk)

        for i in range(0, noise.shape[0], chunk_size):
            chunk = noise[i : i + chunk_size]
            generated_chunk = pred_step(self.generator, chunk)
            generated_chunks.append(generated_chunk)

        # Concatenate all chunks
        generated_images = jnp.concatenate(generated_chunks, axis=0)
        return self._convert_output(generated_images, output_format)

    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        _, state_g = nnx.split(self.generator)
        with open(os.path.join(path, "generator.pkl"), "wb") as f:
            pickle.dump(state_g, f)
        _, state_d = nnx.split(self.discriminator)
        with open(os.path.join(path, "discriminator.pkl"), "wb") as f:
            pickle.dump(state_d, f)
        print(f"GAN saved to {path}")

    def load(self, path: str):
        try:
            with open(os.path.join(path, "generator.pkl"), "rb") as f:
                state_g = pickle.load(f)
            nnx.update(self.generator, state_g)
            with open(os.path.join(path, "discriminator.pkl"), "rb") as f:
                state_d = pickle.load(f)
            nnx.update(self.discriminator, state_d)
            self.optimizer_g = nnx.Optimizer(
                self.generator, optax.adam(0.0002, b1=0.5, b2=0.999)
            )
            self.optimizer_d = nnx.Optimizer(
                self.discriminator, optax.adam(0.0002, b1=0.5, b2=0.999)
            )
            print(f"GAN loaded from {path}")
        except Exception as e:
            raise Exception(f"Error loading GAN: {e}")
