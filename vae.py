"""
VAE and CVAE classes

Written by Christoph Scherer, TU Berlin, 2024

References:
1. Bishop, C. M. & Bishop, H. Deep Learning: Foundations and Concepts. (Springer International Publishing, Cham, 2024). doi:10.1007/978-3-031-45468-4.
2. Doersch, C. Tutorial on Variational Autoencoders. Preprint at https://doi.org/10.48550/arXiv.1606.05908 (2021).
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint

from generative_model import GenerativeModel

def give_weights(n_epochs: int, start: float = 0.5, stop: float = 0.9, final_ratio: float = 0.1, normalize: bool = False) -> torch.Tensor:
    """gives weights for KL annealing schedule (beta annealing)

    - idea: S. R. Bowman, L. Vilnis, O. Vinyals, A. M. Dai, R. Jozefowicz, und S. Bengio, „Generating Sentences from a Continuous Space“. arXiv, 12. Mai 2016. Zugegriffen: 27. März 2024. [Online]. Verfügbar unter: http://arxiv.org/abs/1511.06349
    - my idea: normalize weights

    Args:
        n_epochs (int): one set of weight per epoch
        start (float): start in percent of whole training schedule to start beta annealing
        stop (float): end in percent of whole training schedule to stop beta annealing
        final_ratio (float): final ratio 
        normalize (bool, optional): normalize weights s.t. w_1**2 + w_2**2 = 1. Defaults to False.

    Returns:
        torch.Tensor: stacked weights
    """
    x = torch.linspace(0, 1, n_epochs)
    
    # Determine the interval and offset for the logistic function
    dx = stop - start
    offset = start + dx / 2
    k = 10 / dx  # Logistic growth rate
    
    # Calculate the initial weights using a sigmoid function
    w_1 = torch.sigmoid(k * (x - offset))
    
    # Calculate w_2 as a constant 1 initially (we will adjust it later)
    w_2 = torch.ones_like(w_1)
    
    # Enforce the final ratio w1/w2 at the last epoch
    w_1_final = final_ratio * w_2[-1]  # Since w_2 is initially 1, w_2[-1] = 1
    scaling_factor = w_1_final / w_1[-1]
    
    # Scale the entire w_1 sequence
    w_1 *= scaling_factor
    
    # If normalization is required, normalize w_1 and w_2
    if normalize:
        norm_factor = torch.sqrt(w_1**2 + w_2**2)
        w_1 /= norm_factor
        w_2 /= norm_factor
    
    # Stack the weights and return them as a tensor
    w = torch.stack([w_1, w_2], dim=1)
    return w

class Encoder(nn.Module):
    """
    Encoder submodule for VAE
    """
    def __init__(self, device, n_input_features: int, n_latent_dims: int = 8, n_hidden_encode: int = 4, n_hidden_layers_encode: int = 3):
        super().__init__()
        self.device = device
        self.flatten = nn.Flatten().to(self.device)

        layers = [nn.Linear(n_input_features, n_hidden_encode)] 
        for _ in range(n_hidden_layers_encode):
            linear_layer = nn.Linear(n_hidden_encode, n_hidden_encode)
            nn.init.kaiming_uniform_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(nn.LeakyReLU(0.2),)
        self.encoder_hidden = nn.Sequential(*layers).to(self.device)
        self.encoder_mean = nn.Sequential(
        # linear mapping for mean according to bishop2023learning p. 573
            nn.Linear(int(n_hidden_encode), n_latent_dims), # maps hidden layer to latent means
            #nn.ReLU(),
        ).to(self.device)
        self.encoder_log_var = nn.Sequential(
        # non-linear mapping for log variance
            nn.Linear(int(n_hidden_encode), n_latent_dims), # maps hidden layer to latent standard deviation
            #nn.ReLU(),
        ).to(self.device)

        print("encoder device: " + str(self.device))

    def get_latent_features(self, x):
        x = self.flatten(x).to(self.device)
        # map to hidden layer
        x_hidden = self.encoder_hidden(x.to(self.device)).to(self.device)
        # map to latent space layers
        z_mean = self.encoder_mean(x_hidden.to(self.device)).to(self.device)
        ############################
        z_log_var = self.encoder_log_var(x_hidden.to(self.device)).to(self.device) # exponential non-linear mapping for std according to bishop2023learning p. 573
        ############################
        return z_mean, z_log_var

    def forward(self, x):
        x = torch.flatten(x, start_dim=1).float().to(self.device)
        # get latent space features
        z_mean, z_log_var = self.get_latent_features(x)
        # sample gaussian noise
        eps = torch.randn(z_mean.shape).to(self.device)
        # reparameterization trick: see bishop2023learning p. 575 19.18
        z = z_mean + eps * torch.exp(0.5*z_log_var)
        return z, z_mean, z_log_var
    
class Decoder(nn.Module):
    """
    Decoder submodule for VAE
    """
    def __init__(self, device, n_output_features: int, n_latent_dims: int = 8, n_hidden_decode: int = 4, n_hidden_layers_decode: int = 3):
        super().__init__()

        self.device = device
        self.unflatten = nn.Unflatten(1, torch.Size([int(n_output_features)]))
        
        layers = [nn.Linear(n_latent_dims, n_hidden_decode)] 
        for _ in range(n_hidden_layers_decode):
            linear_layer = nn.Linear(n_hidden_decode, n_hidden_decode)
            nn.init.kaiming_uniform_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(nn.LeakyReLU(0.2),)
        layers.append(nn.Linear(n_hidden_decode, n_output_features))
        self.decoder = nn.Sequential(*layers).to(self.device)

        print("decoder device: " + str(self.device))

    def forward(self, z):
        z = z.float().to(self.device)
        x_recon = self.decoder(z) # decode
        x_recon = self.unflatten(x_recon) # reshape to original shape of x
        return x_recon
    
class VAE(GenerativeModel):
    """
    VAE class without conditioning
    """
    def __init__(self, n_point_features: int, n_label_features: int, n_latent_dims: int = 8, n_hidden_encode: int = 64, n_hidden_decode: int = 64, n_hidden_layers_encode: int = 3, n_hidden_layers_decode: int = 3, device=None):
        super().__init__()
        if device is None:
            # Get cpu, gpu or mps device for training.
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            print(f"Using {self.device} device")
        else: 
            self.device = device
        
        self.n_point_features = n_point_features
        self.n_label_features = n_label_features

        self.n_latent_dims = n_latent_dims
        self.n_hidden_encode = n_hidden_encode
        self.n_hidden_decode = n_hidden_decode

        self.encoder = Encoder(device=self.device, n_input_features=n_point_features, n_latent_dims=n_latent_dims, n_hidden_encode=n_hidden_encode, n_hidden_layers_encode=n_hidden_layers_encode).to(self.device)
        self.decoder = Decoder(device=self.device, n_output_features=n_point_features, n_latent_dims=n_latent_dims, n_hidden_decode=n_hidden_decode, n_hidden_layers_decode=n_hidden_layers_decode).to(self.device)

        
    def forward(self, x, y=None):
        x = x.to(self.device)
        # encode to latent space
        z, z_mean, z_log_var  = self.encoder(x)
        # get reconstruction p(x|z, w)
        x_recon = self.decoder(z)
        # Compute loss
        return x_recon, z, z_mean, z_log_var

        
    def loss_fn(self, x, x_recon, z_mean, z_log_var, weights=torch.Tensor([0.000,1])):
        # loss function
        # calculating loss acc to bishop2023learning p. 577 Alg. 19.1
        x = torch.flatten(x, start_dim=1).to(self.device)
        L_KL = weights[0] * -.5 * torch.mean(torch.sum((1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var)), dim=1)) # compress to scalar, dim=1 for summation only over latent dimensions
        L_recon = weights[1] * torch.nn.functional.mse_loss(x_recon, x, reduction = 'mean')
        #L_recon = weights[1] * torch.nn.functional.binary_cross_entropy(x_recon, x, reduction = 'mean')
        loss = (L_KL + L_recon).float()
        loss_terms = torch.Tensor([L_KL, L_recon])
        return loss, loss_terms

    

    def train(self, training_loader, n_epochs: int = 10, print_training=True, weight_schedule=None, learning_rate=1e-3) -> tuple[np.ndarray, np.ndarray]:
        #size = len(training_loader.dataset)
        self.encoder.train()
        self.decoder.train()
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate)

        losses = []
        mean_losses = []
        loss_terms_list = []
        mean_loss_terms = []
        
        for epoch in range(n_epochs):
            for batch, (x, y) in enumerate(training_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                # encode to latent space
                x_recon, z, z_mean, z_log_var  = self.forward(x, y)
                # Compute loss
                if weight_schedule is None:
                    loss, loss_terms = self.loss_fn(x, x_recon, z_mean, z_log_var)
                else:
                    weights = weight_schedule[epoch, :] # extract weights for this epoch
                    loss, loss_terms = self.loss_fn(x, x_recon, z_mean, z_log_var, weights)

                # Backpropagation
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                losses.append(loss.detach().cpu().numpy())
                loss_terms_list.append(loss_terms.detach().cpu().numpy())

            if print_training:
                if n_epochs > 50_000:
                    if (epoch+1) % 5_000 == 0:
                        # combined loss
                        mean_loss = np.mean(np.array(losses))
                        losses = []
                        mean_losses.append(mean_loss)
                        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))
                        # seperate loss terms
                        mean_loss_terms.append(np.mean(np.array(loss_terms_list), axis=0))
                        loss_terms_list = []
                elif n_epochs >= 10:
                    if (epoch+1) % int(n_epochs/10) == 0:
                        mean_loss = np.mean(np.array(losses))
                        losses = []
                        mean_losses.append(mean_loss)
                        print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))
                        # seperate loss terms
                        mean_loss_terms.append(np.mean(np.array(loss_terms_list), axis=0))
                        loss_terms_list = []
                else:
                    mean_loss = np.mean(np.array(losses))
                    losses = []
                    mean_losses.append(mean_loss)
                    print("Epoch %d,\t Loss %f " % (epoch+1, mean_loss))
                    # seperate loss terms
                    mean_loss_terms.append(np.mean(np.array(loss_terms_list), axis=0))
                    loss_terms_list = []
        self.encoder.eval()
        self.decoder.eval()
        return np.array(mean_losses), np.array(mean_loss_terms)
    
    def train_ignite(self, training_loader: DataLoader, validation_loader: DataLoader, patience: int=10, n_epochs: int= 10000, print_training=True, weight_schedule=None, learning_rate=1e-3, save_dir:str=None, early_stopping:bool=False):
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=learning_rate)

        self.mean_training_losses = []
        self.mean_validation_losses = []

        self.training_losses = []
        self.validation_losses = []

        def train_step(engine, batch):
            # define one training step
            self.encoder.train()
            self.decoder.train()
            self.optimizer.zero_grad()
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            x_recon, z, z_mean, z_log_var  = self.forward(x, y)
            # Compute loss
            if weight_schedule is None:
                loss, loss_terms = self.loss_fn(x, x_recon, z_mean, z_log_var)
            else:
                weights = weight_schedule[engine.state.epoch-1, :] # extract weights for this epoch
                loss, loss_terms = self.loss_fn(x, x_recon, z_mean, z_log_var, weights)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.training_losses.append(loss.item())
            return loss.item()

        self.trainer = Engine(train_step)


        def validation_step(engine, batch):
            # define one validation step
            self.encoder.eval()
            self.decoder.eval()
            with torch.no_grad():
                x, y = batch
                x_recon, z, z_mean, z_log_var  = self.forward(x, y)
                # Compute loss
                if weight_schedule is None:
                    loss, loss_terms = self.loss_fn(x, x_recon, z_mean, z_log_var)
                else:
                    weights = weight_schedule[engine.state.epoch, :] # extract weights for this epoch
                    loss, loss_terms = self.loss_fn(x, x_recon, z_mean, z_log_var, weights)
            self.validation_losses.append(loss.item())
            return loss.item()
            
        self.evaluator = Engine(validation_step)

        
        def score_function(engine):
            # lower validation loss means higher score
            val_loss = engine.state.output
            return -val_loss

        # enable early stopping
        if early_stopping:
            early_stopping_handler = EarlyStopping(patience=patience, score_function=score_function, trainer=self.trainer)
            # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
            self.evaluator.add_event_handler(Events.COMPLETED, early_stopping_handler)
            
        # save checkpoints
        if save_dir:
            to_save = {'model': self}
            self.checkpoint_handler = ModelCheckpoint(dirname=f'{save_dir}/checkpoints',
                                                      n_saved=1, 
                                                      filename_prefix='vae',
                                                      score_function=score_function,
                                                      require_empty=False)
            # add Checkpoint handler
            self.evaluator.add_event_handler(Events.COMPLETED, self.checkpoint_handler, to_save)
            
            self.final_checkpoint_handler = ModelCheckpoint(dirname=f'{save_dir}/final', 
                                                      filename_prefix='vae_final',
                                                      score_function=score_function,
                                                      require_empty=False)
            # add final checkpoint handler when trainer is finished
            self.trainer.add_event_handler(Events.COMPLETED, self.final_checkpoint_handler, to_save)
            


        def run_validation(engine):
            # validate and keep track of losses
            self.evaluator.run(validation_loader)
            
            
            mean_training_loss = np.mean(self.training_losses)
            mean_validation_loss = np.mean(self.validation_losses)
            #mean_training_loss = engine.state.output
            #mean_validation_loss = self.evaluator.state.output
            
            self.mean_training_losses.append(mean_training_loss)
            self.mean_validation_losses.append(mean_validation_loss)
            # reset losses
            self.training_losses = []
            self.validation_losses = []
            if print_training:
                print("Epoch %d,\t Training Loss %f,\t Validation Loss %f " % (engine.state.epoch, mean_training_loss, mean_validation_loss))

        validate_every = int(np.min([20, int(np.ceil(n_epochs/20))]))
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=validate_every), run_validation)

        
        # run actual training
        self.trainer.run(training_loader, max_epochs=n_epochs)
        self.encoder.eval()
        self.decoder.eval()

        return self, min(self.mean_validation_losses)




    def sample(self, n_samples: int = 50, y = None, back_transformation = None):
        with torch.no_grad():
            z = torch.randn((n_samples, self.n_latent_dims)).to(self.device)
            samples = self.decoder(z).detach()
            if back_transformation:
                samples = back_transformation(samples)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return samples
    
class CVAE(VAE):
    """
    VAE subclass with conditioning (conditional variational autoencoder)
    """
    def __init__(self, n_point_features: int, n_label_features: int, n_latent_dims: int = 8, n_hidden_encode: int = 64, n_hidden_decode: int = 64, n_hidden_layers_encode: int = 3, n_hidden_layers_decode: int = 3, device=None):
        super().__init__(n_point_features=n_point_features, n_label_features=n_label_features, n_latent_dims=n_latent_dims, n_hidden_encode=n_hidden_encode, n_hidden_decode=n_hidden_decode, n_hidden_layers_encode=n_hidden_layers_encode, n_hidden_layers_decode=n_hidden_layers_decode, device=device)

        self.n_input_features = n_point_features + n_label_features # input conditioning by concatenating analagous to the diffusion model, see p.15, fig 6 of C. Doersch, „Tutorial on Variational Autoencoders“. arXiv, 3. Januar 2021. doi: 10.48550/arXiv.1606.05908.

        # mapping the conditioning to latent space
        layers = [nn.Linear(n_label_features, n_hidden_encode)] 
        for _ in range(n_hidden_layers_encode):
            linear_layer = nn.Linear(n_hidden_encode, n_hidden_encode)
            nn.init.kaiming_uniform_(linear_layer.weight)
            layers.append(linear_layer)
            layers.append(nn.LeakyReLU(0.2),)
        layers.append(nn.Linear(n_hidden_encode, n_latent_dims))
        self.conditioning_mapping = nn.Sequential(*layers).to(self.device)

        self.encoder = Encoder(device=self.device, n_input_features=self.n_input_features, n_latent_dims=n_latent_dims, n_hidden_encode=n_hidden_encode, n_hidden_layers_encode=n_hidden_layers_encode).to(self.device)
        self.decoder = Decoder(device=self.device, n_output_features=self.n_point_features, n_latent_dims=n_latent_dims, n_hidden_decode=n_hidden_decode, n_hidden_layers_decode=n_hidden_layers_decode).to(self.device)

    def condition_z(self, z, y):
        # map conditioning to latent space
        y_mapped = self.conditioning_mapping(y)
        # add embedding of label to latent space vector, basically giving the location in latent space to sample from
        z_conditioned = z + y_mapped
        return z_conditioned

    def forward(self, x, y):
        # append label vector to point vector
        if len(x.shape) == 1:
            x = torch.unflatten(x, 0, (-1,self.n_point_features)) # unflatten so cat with labels works correctly
        if len(y.shape) == 1:
            y = torch.unflatten(y, 0, (-1,self.n_label_features)) # unflatten so cat with labels works correctly
        x = x.to(self.device)
        y = y.to(self.device)
        in_vec = torch.cat([x,y], dim=1).to(self.device)
        # encode to latent space
        z, z_mean, z_log_var  = self.encoder(in_vec)
        z_conditioned = self.condition_z(z, y).to(self.device)
        # get reconstruction p(x|z,y, w)
        x_recon = self.decoder(z_conditioned).to(self.device)
        return x_recon, z_conditioned, z_mean, z_log_var

    def sample(self, n_samples: int = 50, y = None, back_transformation = None, cfg_interpolation: float = 0):
        with torch.no_grad():
            # generative noise
            z = torch.randn((n_samples, self.n_latent_dims)).to(self.device)
            if y is not None:
                y = torch.Tensor(y).to(self.device)
                # condition noise with label
                z = self.condition_z(z, y)
            samples = self.decoder(z).detach()
            if back_transformation:
                samples = back_transformation(samples)
            if self.device == 'cuda':
                torch.cuda.empty_cache()
        return samples
