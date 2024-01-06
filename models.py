import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, latent_dim:int = 18, input_dim=1, hidden_channels:list = [16,64,256,512], conditioned:bool = False, num_classes:int = 10): #input_dim = 784, pq las imagenes son de (-1, 28,28)
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.input_dim = input_dim
        self.conditioned = conditioned
        self.num_classes = num_classes
        
        modules = []
        #encoder
        for hidden in hidden_channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=input_dim, out_channels=hidden, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU()))
            input_dim = hidden
        
        self.conv_2mu = nn.Linear(in_features=hidden_channels[-1]*4, out_features=self.latent_dim)
        self.conv_2sigma = nn.Linear(in_features=hidden_channels[-1]*4, out_features=self.latent_dim)
        self.encoder = nn.Sequential(*modules)
        
        self.decoder_input = nn.Linear(in_features=self.latent_dim, out_features=hidden_channels[-1]*4)
        
        modules = []        
        hidden_channels.reverse()
        
        #decoder
        for i in range(len(hidden_channels) -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels=hidden_channels[i], out_channels=hidden_channels[i+1], kernel_size=3, stride=3, padding=1),
                    nn.BatchNorm2d(hidden_channels[i+1]),
                    nn.ReLU()))
            
            
        self.decoder = nn.Sequential(*modules)
        
        self.final_layer = nn.Sequential(
                            nn.Conv2d(hidden_channels[-1], out_channels= self.input_dim,
                                      kernel_size= 3, padding= 1),
                            nn.Sigmoid())
        
        self.conditioned_layer = nn.Embedding(self.num_classes, self.latent_dim)
        
    def forward(self, x, y=None):
        mu, sigma = self.encode(x)
        z = self.reparametrizacion(mu, sigma)
        x = self.decode(z, y)
        return x
    
    def reparametrizacion(self, mu:torch.Tensor, sigma:torch.Tensor):
        eps = torch.randn_like(sigma)
        return mu + torch.exp(0.5*sigma) * eps
    
    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.conv_2mu(x)
        sigma = self.conv_2sigma(x)
        return mu, sigma
    
    def decode(self, x:torch.Tensor, y:torch.Tensor):
        if self.conditioned:
            y = self.conditioned_layer(y)
            x = x + y
            
        x = self.decoder_input(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x
    
    def sample(self,
               num_samples:int,
               current_device, conditioned=None , **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)
        
        y = conditioned
        
        if self.conditioned and conditioned is None:
            y = torch.randint(low=0, high=self.num_classes, size=(num_samples,))

        z = z.to(current_device)

        samples = self.decode(z, y)
        return samples