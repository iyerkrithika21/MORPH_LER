import matplotlib.pyplot as plt
import torch

from .mermaidlite import compute_warped_image_multiNC


def show_as_grid(phi, linewidth=1):
    data_size = phi.size()[-1]
    axes = plt.gca()
    plot_phi = data_size * phi[:, ::4, ::4].detach().cpu() - 0.5
    # plot_phi = data_size * phi.detach().cpu() - 0.5
    axes.set_xlim([-0.5, phi.size()[-2] - 0.5])
    axes.set_ylim([phi.size()[-1] - 0.5, -0.5])
    plt.plot(plot_phi[1], plot_phi[0], linewidth=linewidth)
    plt.plot(
        plot_phi[1].transpose(0, 1), plot_phi[0].transpose(0, 1), linewidth=linewidth
    )


def visualizeRegistration(net, image_A, image_B, N, path, linewidth=1):

    net(image_A, image_B)
    du = (
        (net.phi_AB_vectorfield[:, :, 1:, :-1] - net.phi_AB_vectorfield[:, :, :-1, :-1])
        .detach()
        .cpu()
    )
    dv = (
        (net.phi_AB_vectorfield[:, :, :-1, 1:] - net.phi_AB_vectorfield[:, :, :-1, :-1])
        .detach()
        .cpu()
    )
    dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]

    plt.figure(dpi=400)
    plt.subplot(3, 2, 1)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])

    plt.title("Image A")
    plt.imshow(image_A[N, 0].cpu())
    plt.subplot(3, 2, 2)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title("Image B Warped")
    plt.imshow(net.warped_image_B.detach().cpu()[N, 0])
    show_as_grid(net.phi_AB_vectorfield[N], linewidth=linewidth)
    plt.subplot(3, 2, 3)
    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title("Image B")
    plt.imshow(image_B[N, 0].cpu())

    plt.subplot(3, 2, 4)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    plt.title("Composition of Transforms")
    show_as_grid(
        (
            compute_warped_image_multiNC(
                net.phi_BA_vectorfield[:, :2],
                net.phi_AB_vectorfield[:, :2],
                net.spacing,
                1,
            )
        )[N],
        linewidth=linewidth,
    )

    plt.subplot(3, 2, 5)
    plt.title("Area form of PhiAB")

    plt.imshow(dA[N])
    plt.colorbar()

    plt.subplot(3, 2, 6)
    plt.title("Difference image")
    plt.imshow(image_A[N, 0].cpu() - net.warped_image_B.detach().cpu()[N, 0])
    if path:
        plt.savefig(path)
        plt.clf()
    print("Diffeomorphism Failures per batch")
    print(torch.sum(dA < 0))


def visualizeRegistrationCompact(net, image_A, image_B, N):

    net(image_A, image_B)

    ax = plt.gca()
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    # plt.title("Image B Warped")
    plt.imshow(net.warped_image_B.detach().cpu()[N, 0])
    show_as_grid(net.phi_AB_vectorfield[N])
