import math
import numpy as np
from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """

    def __init__(self, param_dim: tuple = (1, 3, 1, 1)) -> None:
        super(Actnorm, self).__init__()

        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))

        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x: torch.Tensor) -> tuple:
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.scale.squeeze().data.copy_(x.transpose(0, 1).flatten(1).std(1, False) + 1e-6).view_as(self.scale)
            self.bias.squeeze().data.copy_(x.transpose(0, 1).flatten(1).mean(1)).view_as(self.bias)

            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = -1.0 * torch.sum(self.scale.abs().log()) * x.shape[2] * x.shape[3]

        return (z, logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        x = z * self.scale + self.bias
        logdet = torch.sum(self.scale.abs().log()) * z.shape[2] * z.shape[3]

        return (x, logdet)


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """

    def __init__(self, n_channels: int = 3, lu_factorize: bool = False) -> None:
        super(Invertible1x1Conv, self).__init__()

        self.lu_factorize = lu_factorize

        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            p, l, u = torch.lu_unpack(*w.unsqueeze(0).lu())

            # initialize model parameters
            self.p = nn.Parameter(p.squeeze())
            self.l = nn.Parameter(l.squeeze())
            self.u = nn.Parameter(u.squeeze())

            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())

            self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        else:
            self.w = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> tuple:
        _, C, H, W = x.shape

        if self.lu_factorize:
            sign_s = torch.tensor(self.sign_s).to(x.device)

            l = self.l * self.l_mask + torch.eye(C).to(x.device)
            u = self.u * self.l_mask.T + torch.diag(sign_s * self.log_s.exp())

            self.w = self.p @ l @ u

            logdet = self.log_s.sum() * H * W
        else:
            logdet = torch.slogdet(self.w)[-1] * H * W

        z = F.conv2d(x, self.w.view(C, C, 1, 1))

        return (z, logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        _, C, H, W = z.shape

        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(z.device))
            u = torch.inverse(self.u * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp()))

            w_inv = u @ l @ self.p.inverse()

            logdet = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1] * H * W

        x = F.conv2d(z, w_inv.view(C, C, 1, 1))

        return (x, logdet)


class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """

    def __init__(self, n_channels: int, hidden_channels: int) -> None:
        super(AffineCoupling, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(n_channels // 2, hidden_channels, kernel_size=3, padding=1, bias=False),
            Actnorm(param_dim=(1, hidden_channels, 1, 1))
        )
        self.relu_1 = nn.ReLU()

        self.conv_2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, padding=1, bias=False),
            Actnorm(param_dim=(1, hidden_channels, 1, 1))
        )
        self.relu_2 = nn.ReLU()

        self.conv_3 = nn.Conv2d(hidden_channels, n_channels, kernel_size=3)  # output is split into scale and shift components

        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels, 1, 1))  # learned scale (cf RealNVP sec 4.1 / Glow official code

        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv_3.weight.data.zero_()
        self.conv_3.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> tuple:
        x_1, x_2 = x.chunk(2, 1)  # split along channel dim

        h = self.conv_1(x_2)[0]
        h = self.relu_1(h)
        h = self.conv_2(h)[0]
        h = self.relu_2(h)
        h = self.conv_3(h) * self.log_scale_factor.exp()

        shift = h[:, 0::2, :, :]  # shift; take even channels
        scale = h[:, 1::2, :, :]  # scale; take odd channels
        scale = torch.sigmoid(scale + 2.0)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_1 = scale * x_1 + shift
        z_2 = x_2
        z = torch.cat([z_1, z_2], dim=1)  # concat along channel dim

        logdet = torch.sum(scale.log(), dim=[1, 2, 3])

        return (z, logdet)

    def inverse(self, z: torch.Tensor) -> tuple:
        z_1, z_2 = z.chunk(2, 1)  # split along channel dim

        h = self.conv_1(z_2)[0]
        h = self.relu_1(h)
        h = self.conv_2(h)[0]
        h = self.relu_2(h)
        h = self.conv_3(h) * self.log_scale_factor.exp()

        shift = h[:, 0::2, :, :]  # shift; take even channels
        scale = h[:, 1::2, :, :]  # scale; take odd channels
        scale = torch.sigmoid(scale + 2.0)

        x_1 = (z_1 - shift) / scale
        x_2 = z_2
        x = torch.cat([x_1, x_2], dim=1)  # concat along channel dim

        logdet = -1.0 * torch.sum(scale.log(), dim=[1, 2, 3])

        return (x, logdet)


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """

    def __init__(self) -> None:
        super(Squeeze, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels

        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        x = x.reshape(B, C // 4, 2, 2, H, W)  # factor channel dim
        x = x.permute(0, 1, 4, 2, 5, 3)  # transpose to (B, C//4, H, 2, W, 2)
        x = x.reshape(B, C // 4, 2 * H, 2 * W)  # aggregate channel dim factors into spatial dims

        return x


class Split(nn.Module):
    """ Split layer; cf Glow figure 2 / RealNVP figure 4b
    Based on RealNVP multi-scale architecture: splits an input in half along the channel dim; half the vars are
    directly modeled as Gaussians while the other half undergo further transformations (cf RealNVP figure 4b).
    """

    def __init__(self, n_channels: int) -> None:
        super(Split, self).__init__()

        self.gaussianize = Gaussianize(n_channels // 2)

    def forward(self, x: torch.Tensor) -> tuple:
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim

        z1 = x1
        z2, logdet = self.gaussianize(x1, x2)

        return (z1, z2, logdet)

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        x1 = z1
        x2, logdet = self.gaussianize.inverse(z1, z2)

        x = torch.cat([x1, x2], dim=1)  # cat along channel dim

        return (x, logdet)


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """

    def __init__(self, n_channels: int) -> None:
        super(Gaussianize, self).__init__()

        self.net = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2 * n_channels, 1, 1))  # learned scale (cf RealNVP sec 4.1 / Glow official code

        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple:
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale

        mu = h[:, 0::2, :, :]
        logs = h[:, 1::2, :, :]

        z = (x2 - mu) * torch.exp(-1.0 * logs)  # center and scale; log prob is computed at the model forward
        logdet = -1.0 * torch.sum(logs, dim=[1, 2, 3])

        return (z, logdet)

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        h = self.net(z1) * self.log_scale_factor.exp()

        mu = h[:, 0::2, :, :]
        logs = h[:, 1::2, :, :]

        x = mu + z2 * torch.exp(logs)
        logdet = torch.sum(logs, dim=[1, 2, 3])

        return (x, logdet)


class Preprocess(nn.Module):
    def __init__(self) -> None:
        super(Preprocess, self).__init__()

    def forward(self, x: torch.Tensor) -> tuple:
        logdet = -1.0 * math.log(256) * x[0].numel()  # processing each image dim from [0, 255] to [0,1]; per RealNVP sec 4.1 taken into account

        return (x, logdet)  # center x at 0

    def inverse(self, z: torch.Tensor) -> tuple:
        logdet = math.log(256) * z[0].numel()

        return (z, logdet)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def __init__(self, *args, **kwargs) -> None:
        super(FlowSequential, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> tuple:
        sum_logdets = 0.0

        for module in self:
            x, logdet = module(x)
            sum_logdets = sum_logdets + logdet

        return (x, sum_logdets)

    def inverse(self, z: torch.Tensor) -> tuple:
        sum_logdets = 0.0

        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet

        return (z, sum_logdets)


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """

    def __init__(self, n_channels: int, hidden_channels: int, lu_factorize: bool = False) -> None:
        super(FlowStep, self).__init__(
            Actnorm(param_dim=(1, n_channels, 1, 1)),
            Invertible1x1Conv(n_channels, lu_factorize),
            AffineCoupling(n_channels, hidden_channels)
        )


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """

    def __init__(self, n_channels: int, hidden_channels: int, depth: int, lu_factorize: bool = False) -> None:
        super(FlowLevel, self).__init__()

        # network layers
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4 * n_channels, hidden_channels, lu_factorize) for _ in range(depth)])
        self.split = Split(4 * n_channels)

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        z1, z2, logdet_split = self.split(x)

        logdet = logdet_flowsteps + logdet_split

        return (z1, z2, logdet)

    def inverse(self, z1: torch.Tensor, z2: torch.Tensor) -> tuple:
        x, logdet_split = self.split.inverse(z1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)

        logdet = logdet_flowsteps + logdet_split

        return (x, logdet)


class Glow_Net(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""

    def __init__(self, hidden_channels: int, depth: int, n_levels: int, input_dims: tuple = (3, 64, 64), lu_factorize: bool = False) -> None:
        super(Glow_Net, self).__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * (4 ** (n_levels + 1)) / (2 ** n_levels))  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / (2 ** (n_levels + 1)))  # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)

        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess()

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * (2 ** i), hidden_channels, depth, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels, hidden_channels, lu_factorize) for _ in range(depth)])

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> tuple:
        condition = condition.view(*condition.shape, 1, 1).repeat(1, 1, 8, 8)

        x, sum_logdets = self.preprocess(x)
        # pass through flow
        zs = []
        for m in self.flowlevels:
            x, z, logdet = m(x)

            zs.append(z)
            sum_logdets = sum_logdets + logdet

        x = self.squeeze(x - condition)

        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet

        # gaussianize the final z
        z, logdet = self.gaussianize(torch.zeros_like(z), z)

        zs.append(z)
        sum_logdets = sum_logdets + logdet

        return (zs, sum_logdets)

    def inverse(self, zs: Union[list, None], condition: torch.Tensor, batch_size: int = None, z_std: float = 1.0) -> tuple:
        condition = condition.view(*condition.shape, 1, 1).repeat(1, 1, 8, 8)

        if zs == None:  # if no random numbers are passed, generate new from the base distribution
            assert batch_size is not None, 'Must either specify batch_size or pass a batch of z random numbers.'

            zs = [z_std * torch.tensor(np.random.normal(0, 1, (batch_size, *self.output_dims)), dtype=torch.float).squeeze().to(condition.device)]

        # pass through inverse flow
        z, sum_logdets = self.gaussianize.inverse(torch.zeros_like(zs[-1]), zs[-1])

        x, logdet = self.flowstep.inverse(z)
        sum_logdets = sum_logdets + logdet

        x = self.squeeze.inverse(x) + condition

        for i, m in enumerate(reversed(self.flowlevels)):
            z = z_std * (torch.tensor(np.random.normal(0, 1, x.shape), dtype=torch.float).squeeze().to(condition.device) if len(zs) == 1 else zs[-i - 2])

            x, logdet = m.inverse(x, z)
            sum_logdets = sum_logdets + logdet

        # postprocess
        x, logdet = self.preprocess.inverse(x)
        sum_logdets = sum_logdets + logdet

        return (x, sum_logdets)

    @property
    def base_dist(self) -> D.Distribution:
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x: torch.Tensor, condition: torch.Tensor, bits_per_pixel: bool = False) -> torch.Tensor:
        zs, logdet = self.forward(x, condition)
        log_prob = sum(torch.sum(self.base_dist.log_prob(z), dim=[1, 2, 3]) for z in zs) + logdet

        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())

        return log_prob
