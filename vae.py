import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class VAE(nn.Module):
	def forward(self, x):

		x = self.encode(x)
		z, loss = self.reparameterize(x)
		x_hat = self.decode(z)

		return x_hat, loss
	def encode(self, x):
		if x.ndim == 5:
			inp = rearrange(x, 'b c h w t -> (b t) c h w')
		else:
			inp = x
		
		z = self.encoder(inp)
		
		if x.ndim == 5:
			z = rearrange(z, '(b t) c h w -> b c h w t', t=x.shape[-1])
		
		return z
	def decode(self, z):
		if z.ndim == 5:
			inp = rearrange(z, 'b c h w t -> (b t) c h w')
		else:
			inp = z
		x = self.decoder(inp)
		x = self.last_layer(x)
		
		if z.ndim == 5:
			x = rearrange(x, '(b t) c h w -> b c h w t', t=z.shape[-1])
		return x
	def reparameterize(self, x, sample=True):
		if x.ndim == 5:
			inp = rearrange(x, 'b c h w t -> (b t) c h w')
		else:
			inp = x
		
		mu, logvar = self.reparameterization(inp).chunk(2, dim=1)
		if sample:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			z =  eps.mul(std).add_(mu)
			loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		else:
			z = mu
			loss = 0.0

		if x.ndim == 5:
			z = rearrange(z, '(b t) c h w -> b c h w t', t=x.shape[-1])

		return z, loss
	
	def __init__(self, latent_dim=64, dim=3, scaling=16):
		#Encodes to latent space of size LATENT_DIM SIZE/8 SIZE/8
		super().__init__()

		self.latent_dim = latent_dim

		self.encoder = nn.Sequential(
            nn.Conv2d(dim, scaling, 3, 1, 1), #16 SIZE SIZE

			nn.GELU(),
            nn.Conv2d(scaling, scaling*2, 3, 2, 1), #32 SIZE/2 SIZE/2

			nn.GELU(),
            nn.Conv2d(scaling*2, scaling*4, 3, 2, 1), #64 SIZE/4 SIZE/4

	        nn.GELU(),
            nn.Conv2d(scaling*4, scaling*8, 3, 2, 1), #128 SIZE/8 SIZE/8

	        nn.GELU(),
            nn.Conv2d(scaling*8, scaling*8, 3, 2, 1), #128 SIZE/8 SIZE/8

		)

		self.reparameterization = nn.Sequential(
			nn.GELU(),
			nn.Conv2d(scaling*8, self.latent_dim*2, 3, 1, 1), #LATENT_DIM SIZE/8 SIZE/8
		)

		self.decoder = nn.Sequential(
			nn.Conv2d(self.latent_dim, scaling*8, 3, 1, 1), #128 SIZE/8 SIZE/8

			nn.GELU(),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(scaling*8, scaling*8, 3, 1, 1), #64 SIZE/4 SIZE/4

			nn.GELU(),
			nn.Upsample(scale_factor=2, mode='bilinear'),
			nn.Conv2d(scaling*8, scaling*4, 3, 1, 1), #64 SIZE/4 SIZE/4

			nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(scaling*4, scaling*2, 3, 1, 1), #32 SIZE/2 SIZE/2

			nn.GELU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(scaling*2, scaling, 3, 1, 1), #16 SIZE SIZE


			nn.GELU(),
			
			
		)
		self.last_layer = nn.Conv2d(scaling, dim, 3, 1, 1) #3 SIZE SIZE

		

class Discriminator(nn.Module):
	#Small patch discriminator.
	def forward(self, x):
		x = self.conv0(x)
		x = self.silu(x)
		x = self.conv1(x)
		x = self.silu(x)
		x = self.conv2(x)
		return x
	def __init__(self, chans=3):
		super().__init__()
		self.conv0 = nn.Conv2d(chans, 32, 3, 2, 1)
		self.conv1 = nn.Conv2d(32, 64, 3, 2, 1)
		self.conv2 = nn.Conv2d(64, 128, 3, 2, 1)

		self.silu = nn.GELU()