# Source: https://github.com/paul007pl/VRCNet/blob/main/utils/train_utils.py
import torch
import SimpleITK as sitk
import itk
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.utils
import numpy as np
from .loss import VectorFieldLoss

class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def save_model(path, net, net_d=None):
    if net_d is not None:
        torch.save({'net_state_dict': net.state_dict(),
                    'D_state_dict': net_d.state_dict()}, path)
    else:
        torch.save({'net_state_dict': net.state_dict()}, path)



def show(tensor, alpha=None):
    plt.imshow(torchvision.utils.make_grid(tensor[:6], nrow=3)[0].cpu().detach(), alpha=alpha)
    plt.xticks([])
    plt.yticks([])


def show_gray(tensor, alpha=None, rotate=True):
    if(rotate == True):
        rotated_image = torch.rot90(tensor,  k=-1, dims=(-2, -1))  # Rotate by -90 degrees (counterclockwise)
    else:
        rotated_image = tensor
    plt.imshow(torchvision.utils.make_grid(rotated_image[:6], nrow=3)[0].cpu().detach(), alpha=alpha, cmap='gray')
    plt.xticks([])
    plt.yticks([])





def get_itk_warped_image(image_A, image_B, phi_AB_recon, args, identity_map):
    device = args.device
    moving_itk = itk.GetImageFromArray(np.ascontiguousarray(image_A[0,0,:,:].detach().cpu().numpy().astype(np.float32)))
    atlas_itk = itk.GetImageFromArray(np.ascontiguousarray(image_B[0,0,:,:].detach().cpu().numpy().astype(np.float32)))
    
    
    reconstructed_phi_AB = create_itk_transform(phi_AB_recon.to(device), identity_map.to(device), moving_itk, atlas_itk)
    # Estimate the atlas using the reconstructed phi_BA
    interpolator = itk.LinearInterpolateImageFunction.New(moving_itk)
    estimated_image_from_ae = itk.resample_image_filter(moving_itk, 
                                                        transform=reconstructed_phi_AB,
                                                        interpolator=interpolator,
                                                        use_reference_image=True,
                                                        reference_image=atlas_itk)
    ae_image = torch.from_numpy(itk.GetArrayFromImage(estimated_image_from_ae)).unsqueeze(0).to(device)

    return ae_image





def get_contour_levels(identity_check):
    min_value = (torch.min(identity_check[0]).cpu().detach().numpy())
    max_value = (torch.max(identity_check[0]).cpu().detach().numpy())
    
    num_lines = int(identity_check.shape[-1]/5)
    contour_levels = np.linspace(min_value, max_value, num_lines)

    return contour_levels


def calculate_composition(phi, n):
    """Applies phi_fn n times on the identity_map."""
    # Initialize  loss module
    IconLoss = VectorFieldLoss().to(phi.device) 
    IconLoss.assign_identity_map(input_shape=[1, 2, 160, 160])
    IconLoss.icon(phi, phi)
    phi_composed = IconLoss.identity_map
    for _ in range(n):
        phi_composed = IconLoss.phi_AB(phi_composed)
    return phi_composed - IconLoss.identity_map


def calculate_composition_3D(phi, n):
    """Applies phi_fn n times on the identity_map."""
    # Initialize  loss module
    IconLoss = VectorFieldLoss().to(phi.device) 
    IconLoss.assign_identity_map(input_shape=[1, 3, 160, 192, 224])
    IconLoss.icon(phi, phi)
    phi_composed = IconLoss.identity_map
    for _ in range(n):
        phi_composed = IconLoss.phi_AB(phi_composed)
    return phi_composed - IconLoss.identity_map


def calculate_identity_check(phi_AB, phi_BA):
    # Initialize inverse consistent loss module
    IconLoss = VectorFieldLoss().to(phi_AB.device)
    IconLoss.assign_identity_map(input_shape=[1, 2, 160, 160])
    IconLoss.icon(phi_AB, phi_BA)
    
    return IconLoss.phi_AB(IconLoss.phi_BA(IconLoss.identity_map))




def plot_field(field, image=None, rotate = True):
    """Plots a field with contours, and rotates the plots by -90 degrees."""
    if image is not None:
        # Rotate the image by -90 degrees
        # rotated_image = torch.rot90(image,  k=-1, dims=(-2, -1))  # Rotate by -90 degrees (counterclockwise)
        show_gray(image, alpha=0.8, rotate=rotate)
    if(rotate == True):
        # Rotate the field data
        rotated_field = torch.rot90(field, k=-1, dims=(-2, -1))  # Rotate by -90 degrees
    else:
        rotated_field = field
    # Get contour levels
    contour_levels = get_contour_levels(rotated_field * 159)

    # Plot contours for the rotated field
    plt.contour(
        torchvision.utils.make_grid(rotated_field * 159, nrow=3)[0].cpu().detach(),
        levels=contour_levels
    )
    plt.contour(
        torchvision.utils.make_grid(rotated_field * 159, nrow=3)[1].cpu().detach(),
        levels=contour_levels
    )

    plt.axis('off')






def plot_jacobian(displacement_field):
    # mask = mask.detach().cpu().numpy().astype(np.uint8)
    """Plots the Jacobian determinant of the displacement field."""
    np_displacement_field = displacement_field[0].permute(1, 2, 0).detach().cpu().numpy()
    sitk_displacement_field = sitk.GetImageFromArray(np_displacement_field, isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian_det_np_arr = np.log(sitk.GetArrayViewFromImage(jacobian_det_volume))
    plt.imshow(jacobian_det_np_arr, alpha = 0.6, cmap='RdYlBu')







def resampling_transform(image, shape):
    # print(itk.CenteredTransformInitializer.GetTypes())
    imageType = itk.template(image)[0][itk.template(image)[1]]

    dummy_image = itk.image_from_array(
        np.zeros(tuple(reversed(shape)), dtype=itk.array_from_image(image).dtype)
    )
    if len(shape) == 2:
        transformType = itk.MatrixOffsetTransformBase[itk.D, 2, 2]

    else:
        transformType = itk.VersorRigid3DTransform[itk.D]
    initType = itk.CenteredTransformInitializer[transformType, imageType, imageType]
    initializer = initType.New()
    initializer.SetFixedImage(dummy_image)
    initializer.SetMovingImage(image)
    transform = transformType.New()

    initializer.SetTransform(transform)
    initializer.InitializeTransform()

    if len(shape) == 3:
        transformType = itk.CenteredAffineTransform[itk.D, 3]
        t2 = transformType.New()
        t2.SetCenter(transform.GetCenter())
        t2.SetOffset(transform.GetOffset())
        transform = t2
    m = transform.GetMatrix()
    m_a = itk.array_from_matrix(m)

    input_shape = image.GetLargestPossibleRegion().GetSize()

    for i in range(len(shape)):

        m_a[i, i] = image.GetSpacing()[i] * (input_shape[i] / shape[i])

    m_a = itk.array_from_matrix(image.GetDirection()) @ m_a

    transform.SetMatrix(itk.matrix_from_array(m_a))

    return transform
    
def create_itk_transform(phi, ident, image_A, image_B) -> "itk.CompositeTransform":

    # itk.DeformationFieldTransform expects a displacement field, so we subtract off the identity map.
    disp = (phi - ident)[0].detach().cpu()

    network_shape_list = list(ident.shape[2:])

    dimension = len(network_shape_list)
    
    tr = itk.DisplacementFieldTransform[(itk.D, dimension)].New()

    # We convert the displacement field into an itk Vector Image.
    scale = torch.Tensor(network_shape_list)

    for _ in network_shape_list:
        scale = scale[:, None]
    disp *= scale - 1

    # disp is a shape [3, H, W, D] tensor with vector components in the order [vi, vj, vk]
    disp_itk_format = (
        disp.double()
        .numpy()[list(reversed(range(dimension)))]
        .transpose(list(range(1, dimension + 1)) + [0])
    )
    # disp_itk_format is a shape [H, W, D, 3] array with vector components in the order [vk, vj, vi]
    # as expected by itk.

    itk_disp_field = itk.image_from_array(disp_itk_format, is_vector=True)

    tr.SetDisplacementField(itk_disp_field)

    to_network_space = resampling_transform(image_A, list(reversed(network_shape_list)))

    from_network_space = resampling_transform(
        image_B, list(reversed(network_shape_list))
    ).GetInverseTransform()

    phi_AB_itk = itk.CompositeTransform[itk.D, dimension].New()

    phi_AB_itk.PrependTransform(from_network_space)
    phi_AB_itk.PrependTransform(tr)
    phi_AB_itk.PrependTransform(to_network_space)

    # warp(image_A, phi_AB_itk) is close to image_B

    return phi_AB_itk