import torch
import torch.nn as nn

class Generator(nn.Module):
	def __init__(self, input_dim, output_channels, feature_maps=64):
		"""
		Initializes the Generator with user-defined input dimension and output channels.

		:param input_dim: Dimension of the input noise vector (e.g., 100).
		:param output_channels: Number of channels in the output image (e.g., 3 for RGB images).
		:param feature_maps: Base number of feature maps (default: 64).
		"""
		super(Generator, self).__init__()
		self.net = nn.Sequential(
			# Input: N x input_dim x 1 x 1
			nn.ConvTranspose2d(input_dim, feature_maps * 8, 4, 1, 0),  # Output: N x (feature_maps*8) x 4 x 4
			nn.BatchNorm2d(feature_maps * 8),
			nn.ReLU(True),
			nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1),  # Output: N x (feature_maps*4) x 8 x 8
			nn.BatchNorm2d(feature_maps * 4),
			nn.ReLU(True),
			nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1),  # Output: N x (feature_maps*2) x 16 x 16
			nn.BatchNorm2d(feature_maps * 2),
			nn.ReLU(True),
			nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1),  # Output: N x feature_maps x 32 x 32
			nn.BatchNorm2d(feature_maps),
			nn.ReLU(True),
			nn.ConvTranspose2d(feature_maps, output_channels, 4, 2, 1),  # Output: N x output_channels x 64 x 64
			nn.Tanh()
		)

	def forward(self, x):
		return self.net(x)

import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(self, input_channels, feature_dim):
		super(Discriminator, self).__init__()
		self.model = nn.Sequential(
			nn.Conv2d(input_channels, feature_dim, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(feature_dim * 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(feature_dim * 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(feature_dim * 8),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Flatten(),  # Flatten the tensor before the linear layer
			nn.Linear(feature_dim * 8 * 4 * 4, 1),  # Adjust the input features based on the output size after convolutions
			nn.Sigmoid()  # Sigmoid activation for binary classification
		)

	def forward(self, x):
		return self.model(x)


class GAN:
	def __init__(self, noise_dim, image_channels, feature_maps_g=64, feature_maps_d=64):
		"""
		Initializes the GAN with user-defined parameters for the generator and discriminator.

		:param noise_dim: Dimension of the input noise vector for the generator.
		:param image_channels: Number of channels in the generated and real images.
		:param feature_maps_g: Base number of feature maps for the generator (default: 64).
		:param feature_maps_d: Base number of feature maps for the discriminator (default: 64).
		"""
		self.generator = Generator(noise_dim, image_channels, feature_maps_g)
		self.discriminator = Discriminator(image_channels, feature_maps_d)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.generator.to(self.device)
		self.discriminator.to(self.device)
		self.noise_dim = noise_dim

	def train(self, real_data_loader, epochs, learning_rate):
		"""
		Trains the GAN using the provided real data.

		:param real_data_loader: DataLoader for the real dataset.
		:param epochs: Number of training epochs.
		:param learning_rate: Learning rate for the optimizers.
		"""
		criterion = nn.BCELoss()
		self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
		self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

		for epoch in range(epochs):
			for i, real_images in enumerate(real_data_loader):
				real_images = real_images.to(self.device)
				batch_size = real_images.size(0)

				# Create labels
				real_labels = torch.ones(batch_size, 1).to(self.device)
				fake_labels = torch.zeros(batch_size, 1).to(self.device)

				# Train Discriminator
				self.discriminator.zero_grad()
				outputs = self.discriminator(real_images)
				d_loss_real = criterion(outputs, real_labels)
				d_loss_real.backward()

				noise = torch.randn(batch_size, self.noise_dim, 1, 1).to(self.device)
				fake_images = self.generator(noise)
				outputs = self.discriminator(fake_images.detach())
				d_loss_fake = criterion(outputs, fake_labels)
				d_loss_fake.backward()
				self.optimizer_d.step()

				# Train Generator
				self.generator.zero_grad()
				outputs = self.discriminator(fake_images)
				g_loss = criterion(outputs, real_labels)
				g_loss.backward()
				self.optimizer_g.step()

				if (i+1) % 100 == 0:
					print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(real_data_loader)}], '
						  f'Discriminator Loss: {d_loss_real.item() + d_loss_fake.item()}, '
						  f'Generator Loss: {g_loss.item()}')

	def generate(self, num_images):
		"""
		Generates images using the trained generator.

		:param num_images: Number of images to generate.
		:return: Generated images.
		"""
		self.generator.eval()
		with torch.no_grad():
			noise = torch.randn(num_images, self.noise_dim, 1, 1).to(self.device)
			generated_images = self.generator(noise)
		self.generator.train()
		return generated_images

	def save(self, path):
		"""
		Saves the generator and discriminator models.

		:param path: Path to save the models.
		"""
		torch.save(self.generator.state_dict(), path + '/generator.pth')
		torch.save(self.discriminator.state_dict(), path + '/discriminator.pth')
		torch.save(self.optimizer_g.state_dict(), path + '/optimizer_g.pth')
		torch.save(self.optimizer_d.state_dict(), path + '/optimizer_d.pth')

	def load(self, path):
		"""
		Loads the generator and discriminator models.

		:param path: Path to load the models.
		"""
		self.generator.load_state_dict(torch.load(path + '/generator.pth'))
		self.discriminator.load_state_dict(torch.load(path + '/discriminator.pth'))
		self.optimizer_g.load_state_dict(torch.load(path + '/optimizer_g.pth'))
		self.optimizer_d.load_state_dict(torch.load(path + '/optimizer_d.pth'))

