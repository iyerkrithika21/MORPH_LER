import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from icon_registration.mermaidlite import compute_warped_image_multiNC, identity_map_multiN
from collections import namedtuple
import os
from icon_registration.config import device
import types
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F


epsilon = 1e-6
def KLD(mu, log_sigma):
    log_sigma = log_sigma + epsilon
    kld = -0.5 * (1 + log_sigma - mu.pow(2) - (log_sigma).exp()) 
    return torch.mean(kld)


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    # return log_z - z.pow(2) / 2
    return log_z - torch.matmul(z,z.t())/2



def flips(phi, in_percentage=False):
    if len(phi.size()) == 5:
        a = (phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]).detach()
        b = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]).detach()
        c = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]).detach()

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        if in_percentage:
            return torch.mean((dV < 0).float()) * 100.
        else:
            return torch.sum(dV < 0) / phi.shape[0]
    elif len(phi.size()) == 4:
        du = (phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]).detach()
        dv = (phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]).detach()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        if in_percentage:
            return torch.mean((dA < 0).float()) * 100.
        else:
            return torch.sum(dA < 0) / phi.shape[0]
    elif len(phi.size()) == 3:
        du = (phi[:, :, 1:] - phi[:, :, :-1]).detach()
        if in_percentage:
            return torch.mean((du < 0).float()) * 100.
        else:
            return torch.sum(du < 0) / phi.shape[0]
    else:
        raise ValueError()

class RegistrationModule(nn.Module):
    r"""Base class for icon modules that perform registration.

    A subclass of RegistrationModule should have a forward method that
    takes as input two images image_A and image_B, and returns a python function
    phi_AB that transforms a tensor of coordinates.

    RegistrationModule provides a method as_function that turns a tensor
    representing an image into a python function mapping a tensor of coordinates
    into a tensor of intensities :math:`\mathbb{R}^N \rightarrow \mathbb{R}` .
    Mathematically, this is what an image is anyway.

    After this class is constructed, but before it is used, you _must_ call
    assign_identity_map on it or on one of its parents to define the coordinate
    system associated with input images.

    The contract that a successful registration fulfils is:
    for a tensor of coordinates X, self.as_function(image_A)(phi_AB(X)) ~= self.as_function(image_B)(X)

    ie

    .. math::
        I^A \circ \Phi^{AB} \simeq I^B

    In particular, self.as_function(image_A)(phi_AB(self.identity_map)) ~= image_B
    """

    def __init__(self):
        super().__init__()
        self.downscale_factor = 1

    def as_function(self, image):
        """image is a tensor with shape self.input_shape.
        Returns a python function that maps a tensor of coordinates [batch x N_dimensions x ...]
        into a tensor of intensities.
        """
        

        
        spacing = 1.0/(np.array(image.shape[2:]) - 1)
        return lambda coordinates: compute_warped_image_multiNC(
            image, coordinates, spacing, 1
        )

    def assign_identity_map(self, input_shape, parents_identity_map=None):
        self.input_shape = np.array(input_shape)
        self.input_shape[0] = 1
        self.spacing = 1.0 / (self.input_shape[2::] - 1)

        # if parents_identity_map is not None:
        #    self.identity_map = parents_identity_map
        # else:
        _id = identity_map_multiN(self.input_shape, self.spacing)
        self.register_buffer("identity_map", torch.from_numpy(_id), persistent=False)

        if self.downscale_factor != 1:
            child_shape = np.concatenate(
                [
                    self.input_shape[:2],
                    np.ceil(self.input_shape[2:] / self.downscale_factor).astype(int),
                ]
            )
        else:
            child_shape = self.input_shape
        for child in self.children():
            if isinstance(child, RegistrationModule):
                child.assign_identity_map(
                    child_shape,
                    # None if self.downscale_factor != 1 else self.identity_map,
                )

    def adjust_batch_size(self, size):
        shape = self.input_shape
        shape[0] = size
        self.assign_identity_map(shape)

    def forward(image_A, image_B):

        raise NotImplementedError()


class FunctionFromVectorField(RegistrationModule):
    """
    Wrap an inner neural network 'net' that returns a tensor of displacements
    [B x N x H x W (x D)], into a RegistrationModule that returns a function that
    transforms a tensor of coordinates
    """

    def __init__(self,):
        super().__init__()
        

    def forward(self, tensor_of_displacements):
        
        displacement_field = self.as_function(tensor_of_displacements)

        def transform(coordinates):
            if hasattr(coordinates, "isIdentity") and coordinates.shape == tensor_of_displacements.shape:
                
                return coordinates + tensor_of_displacements

            return coordinates + displacement_field(coordinates)

        return transform



class CalVectorField(RegistrationModule):
    def __init__(self,  ):

        super().__init__()

       
        
    def __call__(self, phi_AB_disp, phi_BA_disp):
        return super().__call__(phi_AB_disp, phi_BA_disp)

    def icon(self, phi_AB_disp, phi_BA_disp):

        

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True
        self.identity_map = self.identity_map.to(phi_AB_disp.device)
        self.phi_AB = FunctionFromVectorField().to(phi_AB_disp.device)(phi_AB_disp)
        self.phi_BA = FunctionFromVectorField().to(phi_AB_disp.device)(phi_BA_disp)
        
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map.to(phi_AB_disp.device))
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map.to(phi_AB_disp.device))
        return self.identity_map, self.phi_AB_vectorfield, self.phi_BA_vectorfield


class VectorFieldLoss(RegistrationModule):
    def __init__(self,  ):

        super().__init__()

       
        
    def __call__(self, phi_AB_disp, phi_BA_disp):
        return super().__call__(phi_AB_disp, phi_BA_disp)

    def icon(self, phi_AB_disp, phi_BA_disp):

        

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True
        self.identity_map = self.identity_map.to(phi_AB_disp.device)
        self.phi_AB = FunctionFromVectorField().to(phi_AB_disp.device)(phi_AB_disp)
        self.phi_BA = FunctionFromVectorField().to(phi_AB_disp.device)(phi_BA_disp)
        
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map.to(phi_AB_disp.device))
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map.to(phi_AB_disp.device))
    

        Iepsilon = (
            self.identity_map.to(phi_AB_disp.device)
            + torch.randn(*self.identity_map.shape).to(phi_AB_disp.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # inverse consistency one way

        approximate_Iepsilon1 = self.phi_AB(self.phi_BA(Iepsilon))

        approximate_Iepsilon2 = self.phi_BA(self.phi_AB(Iepsilon))

        inverse_consistency_loss = torch.mean(
            (Iepsilon - approximate_Iepsilon1) ** 2
        ) + torch.mean((Iepsilon - approximate_Iepsilon2) ** 2)

   
        return inverse_consistency_loss*(phi_AB_disp.shape[-1]-1)

    def compute_gradient_icon_loss(self, phi_AB_disp, phi_BA_disp):

        self.identity_map.isIdentity = True
        self.identity_map = self.identity_map.to(phi_AB_disp.device)
        self.phi_AB = FunctionFromVectorField().to(phi_AB_disp.device)(phi_AB_disp)
        self.phi_BA = FunctionFromVectorField().to(phi_AB_disp.device)(phi_BA_disp)
        
        self.phi_AB_vectorfield = self.phi_AB(self.identity_map.to(phi_AB_disp.device))
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map.to(phi_AB_disp.device))


        Iepsilon = (
            self.identity_map
            + torch.randn(*self.identity_map.shape).to(self.identity_map.device)
            * 1
            / self.identity_map.shape[-1]
        )

        # compute squared Frobenius of Jacobian of icon error

        direction_losses = []

        approximate_Iepsilon = self.phi_AB(self.phi_BA(Iepsilon))

        inverse_consistency_error = Iepsilon - approximate_Iepsilon

        delta = 0.001

        if len(self.identity_map.shape) == 4:
            dx = torch.Tensor([[[[delta]], [[0.0]]]]).to(self.identity_map.device)
            dy = torch.Tensor([[[[0.0]], [[delta]]]]).to(self.identity_map.device)
            direction_vectors = (dx, dy)

        elif len(self.identity_map.shape) == 5:
            dx = torch.Tensor([[[[[delta]]], [[[0.0]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dy = torch.Tensor([[[[[0.0]]], [[[delta]]], [[[0.0]]]]]).to(
                self.identity_map.device
            )
            dz = torch.Tensor([[[[0.0]]], [[[0.0]]], [[[delta]]]]).to(
                self.identity_map.device
            )
            direction_vectors = (dx, dy, dz)
        elif len(self.identity_map.shape) == 3:
            dx = torch.Tensor([[[delta]]]).to(self.identity_map.device)
            direction_vectors = (dx,)

        for d in direction_vectors:
            approximate_Iepsilon_d = self.phi_AB(self.phi_BA(Iepsilon + d))
            inverse_consistency_error_d = Iepsilon + d - approximate_Iepsilon_d
            grad_d_icon_error = (
                inverse_consistency_error - inverse_consistency_error_d
            ) / delta
            direction_losses.append(torch.mean(grad_d_icon_error**2))

        grad_inverse_consistency_loss = sum(direction_losses)

        return grad_inverse_consistency_loss*159


    def siamese_loss(self, phi_AB_disp, phi_BA_disp, phi_AB_disp_GT, phi_BA_disp_GT):

        

        # Tag used elsewhere for optimization.
        # Must be set at beginning of forward b/c not preserved by .cuda() etc
        self.identity_map.isIdentity = True
        self.identity_map = self.identity_map.to(phi_AB_disp.device)
        self.phi_AB = FunctionFromVectorField().to(phi_AB_disp.device)(phi_AB_disp)
        self.phi_BA = FunctionFromVectorField().to(phi_AB_disp.device)(phi_BA_disp)

        self.phi_AB_vectorfield = self.phi_AB(self.identity_map.to(phi_AB_disp.device))
        self.phi_BA_vectorfield = self.phi_BA(self.identity_map.to(phi_AB_disp.device))

        self.phi_AB_GT = FunctionFromVectorField().to(phi_AB_disp.device)(phi_AB_disp_GT)
        self.phi_BA_GT = FunctionFromVectorField().to(phi_AB_disp.device)(phi_BA_disp_GT)

        self.phi_AB_vectorfield_GT = self.phi_AB_GT(self.identity_map.to(phi_AB_disp.device))
        self.phi_BA_vectorfield_GT = self.phi_BA_GT(self.identity_map.to(phi_AB_disp.device))

        Iepsilon = (
            self.identity_map.to(phi_AB_disp.device)
            + torch.randn(*self.identity_map.shape).to(phi_AB_disp.device)
            * 1
            / self.identity_map.shape[-1]
        )

        term1 = self.phi_AB(self.phi_BA_GT(Iepsilon)) - self.phi_AB_GT(self.phi_BA(Iepsilon))

        term2 = self.phi_AB_GT(self.phi_BA(Iepsilon)) - self.phi_AB(self.phi_BA_GT(Iepsilon))
    

        siamese_inverse_consistency_loss = torch.mean((term1) ** 2) + torch.mean((term2) ** 2)

        return siamese_inverse_consistency_loss*159




def calculate_latent_inverse_consistency_loss(z_AB, z_BA):
    # Step 1: Compute dot product for each pair of vectors in the batch
    dot_product = torch.sum(z_AB * z_BA, dim=1)
    
    # Step 2: Compute magnitudes (norms) of each vector in the batch
    mag_ab = torch.norm(z_AB, dim=1)
    mag_ba = torch.norm(z_BA, dim=1)
    
    # Step 3: Compute cosine similarity for each pair
    cos_theta = dot_product / (mag_ab * mag_ba)

    # loss  = (1 + cos(theta))/2
    # when cos = -1 i.e vectors are opposite to each other, then they are inverse cosnsitent 

    loss_angle = (1 + cos_theta)/2 

    loss_angle = loss_angle.mean(dim=0)

    # loss_magnitude = (torch.norm(z_AB,  dim=1) - torch.norm(z_BA, dim = 1))
    # loss_magnitude = torch.abs(loss_magnitude.mean(0))

    # Calculate the loss as the norm of (z_ab + z_ba)
    loss_magnitude = torch.mean(torch.norm(z_AB + z_BA, p=2, dim=1) ** 2)  # Squared L2 norm for stability

    return loss_angle, loss_magnitude