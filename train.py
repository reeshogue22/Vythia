import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from net import ConvTransformer
from dataset import Dataset
from utils import *
from sklearn.model_selection import train_test_split
import atexit
import time
from vae import VAE
from tqdm import tqdm
import torchvision.models as models
import torch.nn as nn

def train_test_data(dataset, test_split):
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_split)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)

class Trainer:
    def __init__(self, nlayers=4,
                       epochs=1,
                       batch_size=1,
                       max_ctx_length=8,
                       lr=1e-5,
                       halting_iteration=4096,
                       dmodel=256,
                       nheads=16,
                       save_every=100):
        
        self.nlayers = nlayers
        self.epochs = epochs
        self.max_ctx_length = max_ctx_length
        self.batch_size = batch_size
        self.halting_iteration = halting_iteration
        self.lr = lr
        print("Loading Nets...")
        self.vae = VAE().eval().to("mps")
        self.vae.load_state_dict(torch.load('../saves/vae_checkpoint.pt')['engine'])
        self.vae = torch.compile(self.vae)
        self.engine = ConvTransformer(num_blocks=nlayers, dmodel=dmodel, nheads=nheads, chan=64, outchan=64).to("mps")
        self.optim = torch.optim.RAdam(self.engine.parameters(), lr=self.lr)

        self.dataset = Dataset(max_ctx_length=max_ctx_length)
        print(get_nparams(self.engine), "params in generator net.")
        self.train_dataset, self.test_dataset = train_test_data(self.dataset, .2)
        self.train_dataloader = D.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_dataloader = D.DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)
        _, self.display, self.surface = init_camera_and_window() 

        # self.schedule = torch.optim.lr_scheduler.OneCycleLR(self.optim, self.lr, len(self.train_dataset))


        self.save_every = save_every
        self.step = 0

    def train(self):
        """Train the model."""
    
        print("Beginning Training...")
        running_engine_loss = 0
        # running_critic_loss = 0

        self.engine.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            self.engine.train()
            bar = tqdm(range(len(self.train_dataloader)))
            for i, (x) in enumerate(self.train_dataloader):
                if i  % self.save_every != 0 or i == 0:
                    engine_loss = self.training_step(x)
                    bar.set_description(f'Loss: {engine_loss:.4f}')

                else:
                    self.validation_step(x, i)
                    self.dream_step(x, i)
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
        self.engine.train()
        x = x.to("mps")

        x, y = x[:, :, :, :, :-1], x[:, :, :, :, 1:]

        self.optim.zero_grad()
        with torch.no_grad():
            x = self.vae.encode(x)
            x, _ = self.vae.reparameterize(x, sample=False)
            y = self.vae.encode(y)
            y, _ = self.vae.reparameterize(y, sample=False)

        next_false, _ = self.engine(x)

        # memory = []
        # memory.append(t0)
        # for t in range(x.shape[-1]-1):
        #     mem_cat = torch.stack(memory, -1)
        #     mem_cat, lp = self.engine(mem_cat)
        #     memory.append(mem_cat[:, :, :, :, -1])
        
        # next_false = torch.stack(memory, -1)

        identity_loss = F.l1_loss(next_false, y)

        gloss = identity_loss
        gloss.backward()
        self.optim.step()
        # self.schedule.step()
        return gloss

    def validation_step(self, x, step):
        """
        One validation step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        x, y = x[:, :, :, :, :-1], x[:, :, :, :, 1:]
        self.engine.eval()
        x, y = x.to("mps"), y.to("mps")
        x = self.vae.encode(x)
        x, _ = self.vae.reparameterize(x, sample=False)
        
        y = self.vae.encode(y)
        y, _ = self.vae.reparameterize(y, sample=False)

        y = self.vae.decode(y)
        y = torch.sigmoid(y)
        
        y_false, lp = self.engine(x)

        y_false = self.vae.decode(y_false)
        y_false = torch.sigmoid(y_false)
        loss = F.mse_loss(y_false, y)

        y_seq = torch.cat([y, y_false], 2)
        for i in y_seq[0].unsqueeze(0).split(1, -1):
            show_tensor(i.cpu().squeeze(-1), self.display, self.surface)
            time.sleep(1./8.)

        return loss.item()

    def dream_step(self, x, step):
        """
        One dream step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        with torch.no_grad():
            x, y = x[:, :, :, :, :-1], x[:, :, :, :, 1:]
            x, y = x.to("mps"), y.to("mps")

            memory = [*self.vae.reparameterize(self.vae.encode(x[:, :, :, :, :1]), sample=False)[0].split(1, -1)]
            for i in range(1, 8):
                mem_cat = torch.cat(memory, -1)
                mem_cat, lp = self.engine(mem_cat)
                memory.append(mem_cat[:, :, :, :, -1].unsqueeze(-1))
                
            for i in memory:
                i = self.vae.decode(i)
                i = torch.sigmoid(i)
                show_tensor(i.cpu().squeeze(-1), self.display, self.surface)
                time.sleep(1./8.)

    def save(self, path='../saves/checkpoint.pt'):
        """Save the model to disk."""
        torch.save({
            'optim':self.optim.state_dict(),
            'engine':self.engine.state_dict(),
            }, path)

    def load(self, path='../saves/checkpoint.pt'):
        """Load the model from disk."""

        checkpoint = torch.load(path, map_location='cpu')
        self.engine.load_state_dict(checkpoint['engine'])
        del checkpoint['engine']
        self.optim.load_state_dict(checkpoint['optim'])
        del checkpoint['optim']
        
if __name__ == '__main__':
    trainer = Trainer()
    # try:
    #     trainer.load('../saves/checkpoint.pt')
    # except:
    #     print("No checkpoint found. Training from scratch...")
    

    trainer.train()

