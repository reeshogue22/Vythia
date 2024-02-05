import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from vae import VAE, Discriminator
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
from tqdm import tqdm

class Trainer:
    def __init__(self, epochs=4,
                       batch_size=8,
                       lr=1e-4,
                       dlr=1e-5,
                       save_every=100,
            ):
        self.epochs = epochs
        self.save_every = save_every
        self.engine = VAE().to('mps')
        self.discriminator = Discriminator().to('mps')
        self.optim = torch.optim.RAdam(self.engine.parameters(), lr=lr)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=dlr)
        print(get_nparams(self.engine), "params in generator net.")
        self.dataset = Dataset(max_ctx_length=0, data_aug=True)
        self.train_dataloader = D.DataLoader(self.dataset, shuffle=True, batch_size=batch_size)
        self.step = 0
        
    def train(self):
        """Train the model."""
    
        print("Beginning Training...")
        self.engine.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            self.engine.train()
            bar = tqdm(range(len(self.train_dataloader)))
            for i, (x) in enumerate(self.train_dataloader):
                if not i % self.save_every == 0:
                    engine_loss = self.training_step(x)
                    bar.set_description(f'Loss: {engine_loss:.4f}')
                else:
                    self.save()
                bar.update(1)

    def training_step(self, x):
        """
        One optimization step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        x = x.to('mps')
        self.optim.zero_grad()
   
        x = x.squeeze(-1)
        y_false, kl = self.engine(x)
        y_false = torch.sigmoid(y_false)
        
        disc = self.discriminator(y_false)
        gloss = torch.abs(y_false - x).sum()
        gloss = gloss / x.shape[0]
        
        disc = torch.mean(-disc)

        gloss = gloss + 1.0 * self.weight(gloss, disc) * disc + 1e-4 * kl
        gloss.backward()
        self.optim.step()
        
        if self.step > 5000:
            self.discriminator_optim.zero_grad()

            disc_real = self.discriminator(x)
            disc_fake = self.discriminator(y_false.detach())
            disc_real = torch.mean(torch.relu(1. - disc_real))
            disc_fake = torch.mean(torch.relu(1. + disc_fake))
            disc_loss = (disc_real + disc_fake) / 2
            disc_loss.backward()
            self.discriminator_optim.step()

        gen_loss = gloss.item()
        self.step += 1

        return gen_loss


    def weight(self, gloss, disc):
        if self.step > 5000:
            gloss_grads = torch.autograd.grad(gloss, self.engine.last_layer.weight, retain_graph=True)[0]
            disc_grads = torch.autograd.grad(disc, self.engine.last_layer.weight, retain_graph=True)[0]
            d_weight = torch.norm(gloss_grads) / (torch.norm(disc_grads) + 1e-6)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        else:
            d_weight = 0

        return d_weight

    def save(self, path='../saves/vae_checkpoint.pt'):
        """Save the model to disk."""
        torch.save({
            'optim':self.optim.state_dict(),
            'engine':self.engine.state_dict(),
            }, path)

    def load(self, path='../saves/vae_checkpoint.pt'):
        """Load the model from disk."""

        checkpoint = torch.load(path, map_location='cpu')
        self.engine.load_state_dict(checkpoint['engine'])
        del checkpoint['engine']
        self.optim.load_state_dict(checkpoint['optim'])
        del checkpoint['optim']
if __name__ == '__main__': 
    trainer = Trainer()
    trainer.train()