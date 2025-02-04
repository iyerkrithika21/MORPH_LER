import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import sys
import icon_registration.networks as networks
import transmorph.transmorph  as transmorph_network
from transmorph.losses import transmorph_loss
from transmorph.transmorph import CONFIGS as CONFIGS_TM_TM
import icon_registration as icon
sys.path.append("../utils/")
from utils.loss import calculate_latent_inverse_consistency_loss, VectorFieldLoss, CalVectorField

activation_dict = {
    "tanh": nn.Tanh(),
    "relu": nn.ReLU(),
    "softplus": nn.Softplus(),
    "elu": nn.ELU(),
    "leakyrelu": nn.LeakyReLU(),
    "elu": nn.ELU(alpha=1.0),
    "selu":nn.SELU(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU()
}

criterion = nn.MSELoss()

# Define the Vanilla U-Net Autoencoder
class VanillaUNetAutoencoder(nn.Module):
    def __init__(self, in_channels, latent_dim, dimensions=2, vae=False, activation_choice = 'leakyrelu'):
        super(VanillaUNetAutoencoder, self).__init__()
        
        self.dimensions = dimensions
        self.activation_choice = activation_choice
        if self.dimensions == 2:
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.MaxPool = nn.MaxPool2d
            self.interpolate_mode = "bilinear"
            self.interpolate = F.interpolate
            self.BatchNorm = nn.BatchNorm2d


            # Calculate the size of the fully connected layer input
            # Input size: 40 x 40 (1/4 resolution), after 5 pooling layers of kernel_size=3, the size will be 2x2x512
            fc_input_size = int((512) * (5*5))
            fc_out_size = int((512) * (5*5))
        
        elif self.dimensions == 3: 
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.MaxPool = nn.MaxPool3d
            self.interpolate_mode = "trilinear"
            self.interpolate = F.interpolate
            self.BatchNorm = nn.BatchNorm3d
            fc_input_size = int((512) * (5*6*7))
            fc_out_size = int((512) * (5*6*7))
        else:
            raise ValueError("Dimensions should be 2 for this model")
        
        # Encoder
        self.enc1 = self.Conv(in_channels, 16, kernel_size=3, padding=1,stride=2)
        self.enc2 = self.Conv(16, 32, kernel_size=3, padding=1,stride=2)
        self.enc3 = self.Conv(32, 64, kernel_size=3, padding=1,stride=2)
        self.enc4 = self.Conv(64, 256, kernel_size=3, padding=1,stride=2)
        self.enc5 = self.Conv(256, 512, kernel_size=3, padding=1,stride=2)
        

        if(vae==False):
            fc_out = latent_dim
        else:
            fc_out = latent_dim*2

        
        
        # Latent vector
        self.fc1 = nn.Linear(fc_input_size, fc_out)
        self.fc2 = nn.Linear(latent_dim, fc_out_size)
        
        # Decoder
        self.dec1 = self.ConvTranspose(512, 256, kernel_size=3, padding=1)
        self.dec2 = self.ConvTranspose(256, 128, kernel_size=3, padding=1)
        self.dec3 = self.ConvTranspose(128, 64, kernel_size=3, padding=1)
        self.dec4 = self.ConvTranspose(64, 32, kernel_size=3, padding=1)
        self.dec5 = self.ConvTranspose(32, 16, kernel_size=3, padding=1)
        self.dec6 = self.ConvTranspose(16, in_channels, kernel_size=3, padding=1)
        
        # batchnorm for decoder
        self.bn1 = self.BatchNorm(256)
        self.bn2 = self.BatchNorm(128)
        self.bn3 = self.BatchNorm(64)
        self.bn4 = self.BatchNorm(32)
        self.bn5 = self.BatchNorm(16)

        # trying the same initailization as GradICON
        torch.nn.init.zeros_(self.dec6.weight)
        torch.nn.init.zeros_(self.dec6.bias)
        self.activation = activation_dict[self.activation_choice]

    def encode(self, x):
        # Encoder
        
        x = self.activation(x)
        x1 = self.activation(self.enc1(x))
        
        x2 = self.activation(self.enc2(x1))
        
        x3 = self.activation(self.enc3(x2))
        x4 = self.activation(self.enc4(x3))
        x5 = self.enc5(x4)
        
        # Flatten and project to latent space
        x5 = x5.view(x5.size(0), -1)
        
        latent = self.fc1(x5)
        return latent
    


    def decode(self, latent, stage1_latent=None):


        x5 = self.fc2(latent)
        # x5 = x5.view(x5.size(0), 512, 2*self.stage, 2*self.stage)  # Reshape for decoding
        if(self.dimensions == 2):

            x5 = x5.view(x5.size(0), 512, 5, 5)  # Reshape for decoding
        elif(self.dimensions == 3):
            x5 = x5.view(x5.size(0), 512, 5, 6, 7)  # Reshape for decoding
        
        # Decoder
        x6 = self.interpolate(self.bn1(self.dec1(self.activation(x5))), scale_factor=2, mode=self.interpolate_mode)

        x7 = self.interpolate(self.bn2(self.dec2(self.activation(x6))), scale_factor=2, mode=self.interpolate_mode)

        x8 = self.interpolate(self.bn3(self.dec3(self.activation(x7))), scale_factor=2, mode=self.interpolate_mode)

        x9 = self.interpolate(self.bn4(self.dec4(self.activation(x8))), scale_factor=2, mode=self.interpolate_mode)

        x10 = self.interpolate(self.bn5(self.dec5(self.activation(x9))), scale_factor=2, mode=self.interpolate_mode)

        x11 = self.dec6(x10)

        return x11

    def forward(self, x, stage1_latent=None):
        latent = self.encode(x)
        recon = self.decode(latent)

        return recon, latent


def build_model_toy(model_type = 'gradicon', lmbda = 0.5):
    # all_loss = self.lmbda * inverse_consistency_loss + similarity_loss
    model = icon.GradientICON(icon.FunctionFromVectorField(networks.tallUNet2(dimension=2)),
            # Our image similarity metric. The last channel of x and y is whether the value is interpolated or extrapolated,
            # which is used by some metrics but not this one
            lambda x, y: criterion(x,y),
            lmbda,)
    return model

def build_model_image(model_type='gradicon', sigma = 4, lmbda = 0.5, dimensions=2):
    
    inner_net = icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimensions))
    for _ in range(3):
        inner_net = icon.TwoStepRegistration(
            icon.DownsampleRegistration(inner_net, dimension=dimensions),
            icon.FunctionFromVectorField(networks.tallUNet2(dimension=dimensions))
        )
    if(model_type =='icon'):
        net = icon.InverseConsistentNet(inner_net, icon.LNCC(sigma=8), lmbda=.5)
    elif (model_type == 'gradicon'):
        net = icon.GradientICON(inner_net, icon.BlurredSSD(sigma=sigma), lmbda=lmbda)
    return net


def build_model_image_transmorph():
    config = CONFIGS_TM_TM['TransMorph']
    net = transmorph_network.TransMorph(config=config)
    return net

class LinearizedDeformationAE(nn.Module):
    def __init__(self, in_channels, latent_dim, dimensions=2, activation_choice = "leakyrelu", max_levels=6):
        super(LinearizedDeformationAE, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.dimensions = dimensions
        self.max_levels = max_levels
        self.activation_choice = activation_choice
        self.unet = VanillaUNetAutoencoder(in_channels=self.in_channels, latent_dim=self.latent_dim, dimensions=self.dimensions, activation_choice = self.activation_choice)
    
    def forward(self, x, current_level=6):
        x_recon, latent_x = self.unet(x) #[2, 3, 160, 192, 224]
        roots = []
        current_level = min(self.max_levels, current_level)

        for i in range(current_level):
            roots.append(self.unet.decode(latent_x/(2**i)))
        return x_recon, roots, latent_x



class CombinedRegistrationLEDA(nn.Module):
    def __init__(self, args):
        super(CombinedRegistrationLEDA, self).__init__()
        self.args = args
        self.model_type = args.model_type
        
        self.lmbda = args.lmbda
        self.latent_dim = args.latent_dim
        if('oasis' in args.dataset_name) and self.model_type=="gradicon":
            self.register_net = build_model_image(model_type=args.model_type, sigma=args.sigma, lmbda=args.lmbda, dimensions = args.dimensions)
        
        elif('oasis' in args.dataset_name) and self.model_type=="transmorph":
            self.register_net = build_model_image_transmorph()
            self.cal_transmorph_loss = transmorph_loss()
        
        else:
            self.register_net = build_model_toy(model_type = args.model_type, lmbda = args.lmbda).to(args.device)
        self.max_levels = args.max_levels
        self.ledautoencoder = LinearizedDeformationAE(in_channels=args.channels, latent_dim=self.latent_dim, dimensions=args.dimensions, activation_choice = args.activation_choice, max_levels=self.max_levels).to(args.device)

    def forward(self, image_A, image_B, current_level):
        
        batch_size = image_A.shape[0]
        device = image_A.device
        VFloss = VectorFieldLoss().to(device)
        
        image_size = [dim for dim in image_A.shape[2:]]
        VFloss.assign_identity_map(input_shape=[batch_size, self.args.channels] +  image_size)
        

        # registration net to predict deformation field
        # vector field = displacement + identity map
        if self.model_type=="gradicon":
            registrtaion_loss = self.register_net(image_A, image_B)
            phi_AB_disp = self.register_net.phi_AB_vectorfield - self.register_net.identity_map
            phi_BA_disp = self.register_net.phi_BA_vectorfield - self.register_net.identity_map
        

        elif self.model_type=="transmorph":
            warped_image_A, phi_AB_disp = self.register_net(image_A, image_B, warp_A = False)
            warped_image_B, phi_BA_disp = self.register_net(image_B, image_A, warp_A = False)

            registrtaion_loss = self.cal_transmorph_loss(warped_image_A, image_B, phi_AB_disp, warped_image_B, image_A, phi_BA_disp)
            
        # LEDA for predicting roots and linearizing latent space
        phi_AB_disp_recon, phi_AB_roots, latent_phi_AB = self.ledautoencoder(phi_AB_disp, current_level)
        phi_BA_disp_recon, phi_BA_roots, latent_phi_BA = self.ledautoencoder(phi_BA_disp, current_level)
       
        # calcualte loss for latent space
        loss_angle, loss_magnitude = calculate_latent_inverse_consistency_loss(latent_phi_AB, latent_phi_BA)
        latent_loss = loss_angle + loss_magnitude

        # loss for reconstruction, root composition, and inverse consistency
        mse = 0
        inverse_consistency_loss = 0
        
        # original phi MSE
        mse_loss = criterion(phi_AB_disp_recon, phi_AB_disp)*batch_size +\
                   criterion(phi_BA_disp_recon, phi_BA_disp)*batch_size
        
        # inverse consistency  
        inverse_consistency_loss = VFloss.icon(phi_AB_disp_recon, phi_BA_disp_recon)
        


        for i in range(current_level):
            VFloss.icon(phi_AB_roots[i],phi_BA_roots[i])
            phi_AB_comp = VFloss.identity_map
            phi_BA_comp = VFloss.identity_map
            for _ in range(2**i):
                phi_AB_comp = VFloss.phi_AB(VFloss.phi_AB(phi_AB_comp))
                phi_BA_comp = VFloss.phi_BA(VFloss.phi_BA(phi_BA_comp))

            phi_AB_composed = phi_AB_comp - VFloss.identity_map
            phi_BA_composed = phi_BA_comp - VFloss.identity_map

            mse_loss += criterion(phi_AB_composed, phi_AB_disp)*batch_size +\
                        criterion(phi_BA_composed, phi_BA_disp)*batch_size
            
            inverse_consistency_loss += VFloss.icon(phi_AB_composed, phi_BA_composed)
            

        leda_results = {
                        'mse_loss': mse_loss,
                        'inverse_consistency_loss': inverse_consistency_loss,
                        'latent_loss': latent_loss
        }



        results = {
                    'registration_loss': registrtaion_loss,
                    'leda_results': leda_results
        }

        return results 

    def predict(self, image_A, image_B, current_level):
        # registration net to predict deformation field
        batch_size = image_A.shape[0]
        device = image_A.device
        VFloss = VectorFieldLoss().to(device)
        
        image_size = [dim for dim in image_A.shape[2:]]
        VFloss.assign_identity_map(input_shape=[batch_size, self.args.channels] +  image_size)
        
        if self.model_type == "gradicon":
            registrtaion_loss = self.register_net(image_A, image_B)

            # get the displacement fields from vectorfields
            phi_AB_disp = self.register_net.phi_AB_vectorfield - self.register_net.identity_map
            phi_BA_disp = self.register_net.phi_BA_vectorfield - self.register_net.identity_map
        
        elif self.model_type == "transmorph":
            output_ab = self.register_net(image_A, image_B, warp_A = True)
            output_ba = self.register_net(image_B, image_A, warp_A = False)
            phi_AB_disp = output_ab[1]
            phi_BA_disp = output_ba[1]
        
        VFloss.icon(phi_AB_disp, phi_BA_disp)
        identity_map, phi_AB_vectorfield, phi_BA_vectorfield = VFloss.identity_map, VFloss.phi_AB_vectorfield, VFloss.phi_BA_vectorfield
            
        # LEDA for predicting roots and linearizing latent space
        phi_AB_disp_recon, phi_AB_roots, latent_phi_AB = self.ledautoencoder(phi_AB_disp, current_level)
        phi_BA_disp_recon, phi_BA_roots, latent_phi_BA = self.ledautoencoder(phi_BA_disp, current_level)


        leda_results = {
                        'phi_AB_roots': phi_AB_roots,
                        "phi_BA_roots": phi_BA_roots,
                        'phi_AB_disp_recon': phi_AB_disp_recon,
                        'phi_BA_disp_recon': phi_BA_disp_recon,
                        'latent_phi_AB': latent_phi_AB,
                        'latent_phi_BA': latent_phi_BA,
                       
        }



        results = {
                    'phi_AB_vectorfield': phi_AB_vectorfield,
                    'phi_BA_vectorfield': phi_BA_vectorfield,
                    'identity_map': identity_map,
                    'phi_AB_disp': phi_AB_disp,
                    'phi_BA_disp': phi_BA_disp,
                    'leda_results': leda_results



        }

        return results 



class BaselineModels(nn.Module):
    """Baseline model"""
    def __init__(self, args):
        super(BaselineModels, self).__init__()
        
        self.args = args
        self.model_type = args.model_type
        
        self.lmbda = args.reg_loss_weight
        if('oasis' in args.dataset_name) and self.model_type=="gradicon":
            self.register_net = build_model_image(model_type=args.model_type, sigma=args.sigma, lmbda=self.lmbda, dimensions = args.dimensions)
        elif('oasis' in args.dataset_name) and self.model_type=="transmorph":
            self.register_net = build_model_image_transmorph()
            self.cal_transmorph_loss = transmorph_loss()
        



    def forward(self, image_A, image_B ):

        
        
        batch_size = image_A.shape[0]
        device = image_A.device
        VFloss = VectorFieldLoss().to(device)
        
        image_size = [dim for dim in image_A.shape[2:]]
        VFloss.assign_identity_map(input_shape=[batch_size, self.args.channels] +  image_size)
        

        # registration net to predict deformation field
        # vector field = displacement + identity map
        if self.model_type=="gradicon":
            registrtaion_loss = self.register_net(image_A, image_B)
            phi_AB_disp = self.register_net.phi_AB_vectorfield - self.register_net.identity_map
            phi_BA_disp = self.register_net.phi_BA_vectorfield - self.register_net.identity_map
        elif self.model_type == "transmorph":
            output_ab = self.register_net(image_A, image_B, warp_A = True)
            output_ba = self.register_net(image_B, image_A, warp_A = False)
            registrtaion_loss = self.cal_transmorph_loss(output_ab[0], image_B, output_ab[1], output_ba[0], image_A, output_ba[1])
            phi_AB_disp = output_ab[1]
            phi_BA_disp = output_ba[1]
        results = {
                    'registration_loss': registrtaion_loss,
        }
        return results


    def predict(self, image_A, image_B):
        batch_size = image_A.shape[0]
        device = image_A.device
        VFloss = CalVectorField().to(device)
        
        image_size = [dim for dim in image_A.shape[2:]]
        VFloss.assign_identity_map(input_shape=[batch_size, self.args.channels] +  image_size)
        
        # registration net to predict deformation field
        # vector field = displacement + identity map
        if self.model_type=="gradicon":
            registrtaion_loss = self.register_net(image_A, image_B)
            phi_AB_disp = self.register_net.phi_AB_vectorfield - self.register_net.identity_map
            phi_BA_disp = self.register_net.phi_BA_vectorfield - self.register_net.identity_map
        elif self.model_type == "transmorph":
            output_ab = self.register_net(image_A, image_B, warp_A = True)
            output_ba = self.register_net(image_B, image_A, warp_A = False)
            
            phi_AB_disp = output_ab[1]
            phi_BA_disp = output_ba[1]


        identity_map, phi_AB_vectorfield, phi_BA_vectorfield = VFloss.icon(phi_AB_disp, phi_BA_disp)
        
        results = {
                    'phi_AB_vectorfield': phi_AB_vectorfield,
                    'phi_BA_vectorfield': phi_BA_vectorfield,
                    'identity_map': identity_map,
                    'phi_AB_disp': phi_AB_disp,
                    'phi_BA_disp': phi_BA_disp,


        }
        return results