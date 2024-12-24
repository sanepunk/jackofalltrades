from .Regression import (LinearRegression, LogisticRegression, MLPRegressor, RidgeRegression,
						 AdaptiveRegression)
from .Classification import ImageClassification

from .GAN import Generator, Discriminator, GAN

from .VAE import EncoderDecoder, Encoder, Decoder, load_params, save_params, optimizer, update
