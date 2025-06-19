import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
import tensorflow as tf


def load_and_preprocess_image(filename, img_height, img_width):
	# Load an image from a file
	image = tf.io.read_file(filename)
	# Decode the image
	image = tf.image.decode_image(image, channels=3)
	# Resize the image
	# print(image.shape)
	image = tf.image.resize(image, [img_height, img_width]).numpy()
	image = image.reshape((1, img_height, img_width, 3))
	# Normalize the image
	return image

class Encoder(nn.Module):
	@nn.compact
	def __call__(self, inputs):
		x_16_feature_skip = nn.leaky_relu(nn.Conv(features = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(inputs), negative_slope = 0.2)
		x_16_feature_skip = nn.leaky_relu(nn.Conv(features = 16, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_16_feature_skip), negative_slope = 0.2)
		# print(x_16_feature_skip.shape, "pre-pool-16")
		x_pool_16 = nn.max_pool(x_16_feature_skip, window_shape = (2, 2), strides = (2, 2))

		# 		print(x_pool_16.shape, "post-pool-16")
		x_32_feature_skip = nn.leaky_relu(nn.Conv(features = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_pool_16), negative_slope = 0.2)
		x_32_feature_skip = nn.leaky_relu(nn.Conv(features = 32, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_32_feature_skip), negative_slope = 0.2)
		# 		print(x_32_feature_skip.shape, "pre-pool-32")
		x_pool_32 = nn.max_pool(x_32_feature_skip, window_shape = (2, 2), strides = (2, 2))

		# 		print(x_pool_32.shape, "post-pool-32")
		x_64_feature_skip = nn.leaky_relu(nn.Conv(features = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_pool_32), negative_slope = 0.2)
		x_64_feature_skip = nn.leaky_relu(nn.Conv(features = 64, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_64_feature_skip), negative_slope = 0.2)
		# 		print(x_64_feature_skip.shape, "pre-pool-64")
		x_pool_64 = nn.max_pool(x_64_feature_skip, window_shape = (2, 2), strides = (2, 2))

		# 		print(x_pool_64.shape, "post-pool-64")
		x_128_feature_skip = nn.leaky_relu(nn.Conv(features = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_pool_64), negative_slope = 0.2)
		x_128_feature_skip = nn.leaky_relu(nn.Conv(features = 128, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_128_feature_skip), negative_slope = 0.2)
		# 		print(x_128_feature_skip.shape, "pre-pool-128")
		x_pool_128 = nn.max_pool(x_128_feature_skip, window_shape = (2, 2), strides = (2, 2))

		# 		print(x_pool_128.shape, "post-pool-128")
		x_256_feature_skip = nn.leaky_relu(nn.Conv(features = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_pool_128), negative_slope = 0.2)
		x_256_feature_skip = nn.leaky_relu(nn.Conv(features = 256, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_256_feature_skip), negative_slope = 0.2)
		# 		print(x_256_feature_skip.shape, "pre-pool-256")
		x_pool_256 = nn.max_pool(x_256_feature_skip, window_shape = (2, 2), strides = (2, 2))

		# 		print(x_pool_256.shape, "post-pool-256")
		x_pool_256_1 = nn.max_pool(x_pool_256, window_shape = (2, 2), strides = (2, 2))

		mu = jnp.mean(x_pool_256_1)
		key = jax.random.PRNGKey(0)
		key, subkey = jax.random.split(key)
		log_var = jnp.log(jnp.var(x_pool_256_1) ** 2)
		parameterization = mu + jax.random.normal(jax.random.PRNGKey(subkey), x_pool_256_1.shape) * jnp.exp(log_var)
		return parameterization, mu, log_var, [inputs, x_pool_16, x_pool_32, x_pool_64, x_pool_128, x_pool_256]


class Decoder(nn.Module):
	@nn.compact
	def __call__(self, inputs: nn.Module, skip_connections: list[nn.Module], input_channels: int, output_channels: int):
		# print(inputs.shape, "hello")
		x_transpose_1 = nn.ConvTranspose(features = input_channels, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME')(inputs)
		x_added_5 = x_transpose_1 + skip_connections[5]
		x_convoluted = nn.Conv(features = input_channels, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_added_5)
		x_convoluted = nn.relu(x_convoluted)
		x_convoluted = nn.Conv(features = input_channels, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_convoluted)
		x_convoluted = nn.relu(x_convoluted)

		x_transpose_2 = nn.ConvTranspose(features = input_channels // 2, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME')(x_convoluted)
		x_added_4 = x_transpose_2 + skip_connections[4]
		x_convoluted = nn.Conv(features = input_channels // 2, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_added_4)
		x_convoluted = nn.relu(x_convoluted)
		x_convoluted = nn.Conv(features = input_channels // 2, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_convoluted)
		x_convoluted = nn.relu(x_convoluted)

		x_transpose_3 = nn.ConvTranspose(features = input_channels // 4, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME')(x_convoluted)
		x_added_3 = x_transpose_3 + skip_connections[3]
		x_convoluted = nn.Conv(features = input_channels // 4, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_added_3)
		x_convoluted = nn.relu(x_convoluted)
		x_convoluted = nn.Conv(features = input_channels // 4, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_convoluted)
		x_convoluted = nn.relu(x_convoluted)

		x_transpose_4 = nn.ConvTranspose(features = input_channels // 8, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME')(x_convoluted)
		x_added_2 = x_transpose_4 + skip_connections[2]
		x_convoluted = nn.Conv(features = input_channels // 8, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_added_2)
		x_convoluted = nn.relu(x_convoluted)
		x_convoluted = nn.Conv(features = input_channels // 8, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_convoluted)
		x_convoluted = nn.relu(x_convoluted)

		x_transpose_5 = nn.ConvTranspose(features = input_channels // 16, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME')(x_convoluted)
		x_added_1 = x_transpose_5 + skip_connections[1]
		x_convoluted = nn.Conv(features = input_channels // 16, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_added_1)
		x_convoluted = nn.relu(x_convoluted)
		x_convoluted = nn.Conv(features = input_channels // 16, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_convoluted)
		x_convoluted = nn.relu(x_convoluted)

		x_transpose_6 = nn.ConvTranspose(features = output_channels, kernel_size = (3, 3), strides = (2, 2), padding = 'SAME')(x_convoluted)
		x_added_0 = x_transpose_6 + skip_connections[0]
		x_convoluted = nn.Conv(features = output_channels, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_added_0)
		x_convoluted = nn.relu(x_convoluted)
		x_convoluted = nn.Conv(features = output_channels, kernel_size = (3, 3), strides = (1, 1), padding = 'SAME')(x_convoluted)
		x_convoluted = nn.sigmoid(x_convoluted)
		# print(x_convoluted.shape, "post-conv")
		return x_convoluted


class EncoderDecoder(nn.Module):
	@nn.compact
	def __call__(self, input1):
		encoder = Encoder()
		decoder = Decoder()
		latent, mu, log_var, skips = encoder(input1)
		reconstructed = decoder(latent, skips, 256, 3)
		return nn.relu(reconstructed), mu, log_var


def optimizer(learning_rate):
	return optax.adam(learning_rate=learning_rate)

def loss(params, x, y):
	predictions, mu, log_var = EncoderDecoder().apply(params, x)
	return jnp.mean(jnp.square(predictions - y))

def update(params, state, x, y):
	optim = optimizer()
	grads = jax.grad(loss)(params, x, y)
	updates, new_state = optim.update(grads, state)
	new_params = optax.apply_updates(params, updates)
	return new_params, new_state

def save_params(params, state, path):
	jnp.savez(path, params=params, state=state)

def load_params(path):
	loaded = jnp.load(path)
	return loaded['params'], loaded['state']


