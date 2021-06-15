"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import os
import time
import math
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torch.nn.functional import one_hot
from torchvision.utils import save_image, make_grid

from utils import parse, load_data


class Actnorm(nn.Module):
    """ Actnorm layer; cf Glow section 3.1 """

    def __init__(self, param_dim=(1, 3, 1, 1)):
        super(Actnorm, self).__init__()

        self.scale = nn.Parameter(torch.ones(param_dim))
        self.bias = nn.Parameter(torch.zeros(param_dim))

        self.register_buffer('initialized', torch.tensor(0).byte())

    def forward(self, x):
        if not self.initialized:
            # per channel mean and variance where x.shape = (B, C, H, W)
            self.scale.squeeze().data.copy_(x.transpose(0, 1).flatten(1).std(1, False) + 1e-6).view_as(self.scale)
            self.bias.squeeze().data.copy_(x.transpose(0, 1).flatten(1).mean(1)).view_as(self.bias)

            self.initialized += 1

        z = (x - self.bias) / self.scale
        logdet = -1.0 * torch.sum(self.scale.abs().log()) * x.shape[2] * x.shape[3]

        return z, logdet

    def inverse(self, z):
        return z * self.scale + self.bias, torch.sum(self.scale.abs().log()) * z.shape[2] * z.shape[3]


class Invertible1x1Conv(nn.Module):
    """ Invertible 1x1 convolution layer; cf Glow section 3.2 """

    def __init__(self, n_channels=3, lu_factorize=False):
        super(Invertible1x1Conv, self).__init__()

        self.lu_factorize = lu_factorize

        # initiaize a 1x1 convolution weight matrix
        w = torch.randn(n_channels, n_channels)
        w = torch.qr(w)[0]  # note: nn.init.orthogonal_ returns orth matrices with dets +/- 1 which complicates the inverse call below

        if lu_factorize:
            # compute LU factorization
            p, l, u = torch.lu_unpack(*w.unsqueeze(0).lu())
            # initialize model parameters
            self.p, self.l, self.u = nn.Parameter(p.squeeze()), nn.Parameter(l.squeeze()), nn.Parameter(u.squeeze())
            s = self.u.diag()
            self.log_s = nn.Parameter(s.abs().log())
            self.register_buffer('sign_s', s.sign())  # note: not optimizing the sign; det W remains the same sign
            self.register_buffer('l_mask', torch.tril(torch.ones_like(self.l), -1))  # store mask to compute LU in forward/inverse pass
        else:
            self.w = nn.Parameter(w)

    def forward(self, x):
        _, C, H, W = x.shape

        if self.lu_factorize:
            l = self.l * self.l_mask + torch.eye(C).to(self.l.device)
            u = self.u * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp())
            self.w = self.p @ l @ u
            logdet = self.log_s.sum() * H * W
        else:
            logdet = torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(x, self.w.view(C, C, 1, 1)), logdet

    def inverse(self, z):
        _, C, H, W = z.shape

        if self.lu_factorize:
            l = torch.inverse(self.l * self.l_mask + torch.eye(C).to(self.l.device))
            u = torch.inverse(self.u * self.l_mask.T + torch.diag(self.sign_s * self.log_s.exp()))
            w_inv = u @ l @ self.p.inverse()
            logdet = - self.log_s.sum() * H * W
        else:
            w_inv = self.w.inverse()
            logdet = - torch.slogdet(self.w)[-1] * H * W

        return F.conv2d(z, w_inv.view(C, C, 1, 1)), logdet


class AffineCoupling(nn.Module):
    """ Affine coupling layer; cf Glow section 3.3; RealNVP figure 2 """

    def __init__(self, n_channels, width):
        super(AffineCoupling, self).__init__()

        # per realnvp, network splits input, operates on half of it, and returns shift and scale of dim = half the input channels
        self.conv1 = nn.Conv2d(n_channels // 2, width, kernel_size=3, padding=1, bias=False)  # input is split along channel dim
        self.actnorm1 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv2 = nn.Conv2d(width, width, kernel_size=1, padding=1, bias=False)
        self.actnorm2 = Actnorm(param_dim=(1, width, 1, 1))
        self.conv3 = nn.Conv2d(width, n_channels, kernel_size=3)  # output is split into scale and shift components

        self.log_scale_factor = nn.Parameter(torch.zeros(n_channels, 1, 1))  # learned scale (cf RealNVP sec 4.1 / Glow official code

        # initialize last convolution with zeros, such that each affine coupling layer performs an identity function
        self.conv3.weight.data.zero_()
        self.conv3.bias.data.zero_()

    def forward(self, x):
        x_a, x_b = x.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(x_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:, 0::2, :, :]  # shift; take even channels
        s = h[:, 1::2, :, :]  # scale; take odd channels
        s = torch.sigmoid(s + 2.0)  # at initalization, s is 0 and sigmoid(2) is near identity

        z_a = s * x_a + t
        z_b = x_b
        z = torch.cat([z_a, z_b], dim=1)  # concat along channel dim

        logdet = s.log().sum([1, 2, 3])

        return z, logdet

    def inverse(self, z):
        z_a, z_b = z.chunk(2, 1)  # split along channel dim

        h = F.relu(self.actnorm1(self.conv1(z_b))[0])
        h = F.relu(self.actnorm2(self.conv2(h))[0])
        h = self.conv3(h) * self.log_scale_factor.exp()
        t = h[:, 0::2, :, :]  # shift; take even channels
        s = h[:, 1::2, :, :]  # scale; take odd channels
        s = torch.sigmoid(s + 2.0)

        x_a = (z_a - t) / s
        x_b = z_b
        x = torch.cat([x_a, x_b], dim=1)  # concat along channel dim

        logdet = - s.log().sum([1, 2, 3])

        return x, logdet


class Squeeze(nn.Module):
    """ RealNVP squeezing operation layer (cf RealNVP section 3.6; Glow figure 2b):
    For each channel, it divides the image into subsquares of shape 2 × 2 × c, then reshapes them into subsquares of
    shape 1 × 1 × 4c. The squeezing operation transforms an s × s × c tensor into an s × s × 4c tensor """

    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // 2, 2, W // 2, 2)  # factor spatial dim
        x = x.permute(0, 1, 3, 5, 2, 4)  # transpose to (B, C, 2, 2, H//2, W//2)
        x = x.reshape(B, 4 * C, H // 2, W // 2)  # aggregate spatial dim factors into channels
        return x

    def inverse(self, x):
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

    def __init__(self, n_channels):
        super(Split, self).__init__()

        self.gaussianize = Gaussianize(n_channels // 2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # split input along channel dim
        z2, logdet = self.gaussianize(x1, x2)

        return x1, z2, logdet

    def inverse(self, x1, z2):
        x2, logdet = self.gaussianize.inverse(x1, z2)
        x = torch.cat([x1, x2], dim=1)  # cat along channel dim

        return x, logdet


class Gaussianize(nn.Module):
    """ Gaussianization per ReanNVP sec 3.6 / fig 4b -- at each step half the variables are directly modeled as Gaussians.
    Model as Gaussians:
        x2 = z2 * exp(logs) + mu, so x2 ~ N(mu, exp(logs)^2) where mu, logs = f(x1)
    then to recover the random numbers z driving the model:
        z2 = (x2 - mu) * exp(-logs)
    Here f(x1) is a conv layer initialized to identity.
    """

    def __init__(self, n_channels):
        super(Gaussianize, self).__init__()

        self.net = nn.Conv2d(n_channels, 2 * n_channels, kernel_size=3, padding=1)  # computes the parameters of Gaussian
        self.log_scale_factor = nn.Parameter(torch.zeros(2 * n_channels, 1, 1))  # learned scale (cf RealNVP sec 4.1 / Glow official code
        # initialize to identity
        self.net.weight.data.zero_()
        self.net.bias.data.zero_()

    def forward(self, x1, x2):
        h = self.net(x1) * self.log_scale_factor.exp()  # use x1 to model x2 as Gaussians; learnable scale
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]          # split along channel dims
        z2 = (x2 - m) * torch.exp(-logs)                # center and scale; log prob is computed at the model forward
        logdet = - logs.sum([1, 2, 3])

        return z2, logdet

    def inverse(self, x1, z2):
        h = self.net(x1) * self.log_scale_factor.exp()
        m, logs = h[:, 0::2, :, :], h[:, 1::2, :, :]
        x2 = m + z2 * torch.exp(logs)
        logdet = logs.sum([1, 2, 3])

        return x2, logdet


class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x):
        logdet = - math.log(256) * x[0].numel()  # processing each image dim from [0, 255] to [0,1]; per RealNVP sec 4.1 taken into account
        return x - 0.5, logdet  # center x at 0

    def inverse(self, x):
        logdet = math.log(256) * x[0].numel()
        return x + 0.5, logdet


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def __init__(self, *args, **kwargs):
        super(FlowSequential, self).__init__(*args, **kwargs)

    def forward(self, x):
        sum_logdets = 0.0

        for module in self:
            x, logdet = module(x)
            sum_logdets = sum_logdets + logdet

        return x, sum_logdets

    def inverse(self, z):
        sum_logdets = 0.0

        for module in reversed(self):
            z, logdet = module.inverse(z)
            sum_logdets = sum_logdets + logdet

        return z, sum_logdets


class FlowStep(FlowSequential):
    """ One step of Glow flow (Actnorm -> Invertible 1x1 conv -> Affine coupling); cf Glow Figure 2a """

    def __init__(self, n_channels, width, lu_factorize=False):
        super(FlowStep, self).__init__(
            Actnorm(param_dim=(1, n_channels, 1, 1)),
            Invertible1x1Conv(n_channels, lu_factorize),
            AffineCoupling(n_channels, width)
        )


class FlowLevel(nn.Module):
    """ One depth level of Glow flow (Squeeze -> FlowStep x K -> Split); cf Glow figure 2b """

    def __init__(self, n_channels, width, depth, lu_factorize=False):
        super(FlowLevel, self).__init__()
        # network layers
        self.squeeze = Squeeze()
        self.flowsteps = FlowSequential(*[FlowStep(4 * n_channels, width, lu_factorize) for _ in range(depth)])
        self.split = Split(4 * n_channels)

    def forward(self, x):
        x = self.squeeze(x)
        x, logdet_flowsteps = self.flowsteps(x)
        x1, z2, logdet_split = self.split(x)

        logdet = logdet_flowsteps + logdet_split

        return x1, z2, logdet

    def inverse(self, x1, z2):
        x, logdet_split = self.split.inverse(x1, z2)
        x, logdet_flowsteps = self.flowsteps.inverse(x)
        x = self.squeeze.inverse(x)

        logdet = logdet_flowsteps + logdet_split

        return x, logdet


class Glow(nn.Module):
    """ Glow multi-scale architecture with depth of flow K and number of levels L; cf Glow figure 2; section 3"""

    def __init__(self, width, depth, n_levels, input_dims=(3, 64, 64), lu_factorize=False):
        super(Glow, self).__init__()
        # calculate output dims
        in_channels, H, W = input_dims
        out_channels = int(in_channels * (4 ** (n_levels + 1)) / (2 ** n_levels))  # each Squeeze results in 4x in_channels (cf RealNVP section 3.6); each Split in 1/2x in_channels
        out_HW = int(H / (2 ** (n_levels + 1)))  # each Squeeze is 1/2x HW dim (cf RealNVP section 3.6)
        self.output_dims = out_channels, out_HW, out_HW

        # preprocess images
        self.preprocess = Preprocess()

        # network layers cf Glow figure 2b: (Squeeze -> FlowStep x depth -> Split) x n_levels -> Squeeze -> FlowStep x depth
        self.flowlevels = nn.ModuleList([FlowLevel(in_channels * (2 ** i), width, depth, lu_factorize) for i in range(n_levels)])
        self.squeeze = Squeeze()
        self.flowstep = FlowSequential(*[FlowStep(out_channels, width, lu_factorize) for _ in range(depth)])

        # gaussianize the final z output; initialize to identity
        self.gaussianize = Gaussianize(out_channels)

        # base distribution of the flow
        self.register_buffer('base_dist_mean', torch.zeros(1))
        self.register_buffer('base_dist_var', torch.ones(1))

    def forward(self, x, condition):
        zero = torch.zeros(condition.shape[0], 24).to(condition.device)
        condition = torch.cat([condition, zero], dim=1)
        condition = condition.view(condition.shape[0], 1, 8, 8).repeat(1, 24, 1, 1)

        x, sum_logdets = self.preprocess(x)
        # pass through flow
        zs = []
        for m in self.flowlevels:
            x, z, logdet = m(x)
            sum_logdets = sum_logdets + logdet
            zs.append(z)

        x = self.squeeze(x - condition)

        z, logdet = self.flowstep(x)
        sum_logdets = sum_logdets + logdet

        # gaussianize the final z
        z, logdet = self.gaussianize(torch.zeros_like(z), z)
        sum_logdets = sum_logdets + logdet
        zs.append(z)

        return zs, sum_logdets

    def inverse(self, condition, zs=None, batch_size=None, z_std=1.0):
        zero = torch.zeros(condition.shape[0], 24).to(condition.device)
        condition = torch.cat([condition, zero], dim=1)
        condition = condition.view(condition.shape[0], 1, 8, 8).repeat(1, 24, 1, 1)

        if zs is None:  # if no random numbers are passed, generate new from the base distribution
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

        return x, sum_logdets

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def log_prob(self, x, condition, bits_per_pixel=False):
        zs, logdet = self.forward(x, condition)
        log_prob = sum(self.base_dist.log_prob(z).sum([1, 2, 3]) for z in zs) + logdet

        if bits_per_pixel:
            log_prob /= (math.log(2) * x[0].numel())

        return log_prob


@torch.no_grad()
def generate(model, condition, n_samples, z_stds):
    model.eval()

    print('Generating ...', end='\r')

    samples = []
    for z_std in z_stds:
        sample, _ = model.inverse(condition, batch_size=n_samples, z_std=z_std)

        samples.append(sample)

    return torch.cat(samples, 0)


def train(model, train_loader, optimizer, args):
    for epoch in range(args.start_epoch, args.start_epoch + args.n_epochs):
        model.train()

        tic = time.time()
        for i, (image, label) in enumerate(train_loader):
            image = image.to(args.device)
            label = label.to(args.device)

            args.step += args.world_size
            # warmup learning rate
            if epoch <= args.n_epochs_warmup:
                optimizer.param_groups[0]['lr'] = args.lr * min(1, args.step / (len(train_loader) * args.world_size * args.n_epochs_warmup))

            loss = - model.log_prob(image, label, bits_per_pixel=True).mean(0)

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

            optimizer.step()

            et = time.time() - tic
            print(f'Epoch: [{epoch + 1}/{args.start_epoch + args.n_epochs}][{i + 1}/{len(train_loader)}], Time: {(et // 60):.0f}m{(et % 60):.02f}s, Loss: {loss.item():.3f}')

            if (i + 1) % 10 == 0:
                samples = generate(model, label, n_samples=label.shape[0], z_stds=[1.0])

                images = make_grid(samples.cpu(), nrow=4, pad_value=1, normalize=True)
                save_image(images, f'images/generated_sample_{args.step}.png')

                torch.save(model, f'weights/nf/{epoch + 1}.pth')


def _gen_label(size: int, min_object_amount: int = 3, max_object_amount: int = 10, num_classes: int = 40) -> torch.Tensor:
    gen_label = []

    for _ in range(size):
        object_amount = np.random.randint(min_object_amount, max_object_amount, 1)

        temp_gen_label = np.random.choice(range(num_classes), object_amount, replace=False)
        temp_gen_label = one_hot(torch.tensor(temp_gen_label), num_classes)

        gen_label.append(torch.sum(temp_gen_label, dim=0).view(1, -1))

    gen_label = torch.cat(gen_label, dim=0)
    gen_label[gen_label == 0] = -1

    gen_label = gen_label.type(torch.float)

    return gen_label


if __name__ == '__main__':
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cuda.manual_seed_all(seed)

    args = parse()

    train_loader = load_data(args.root_folder)

    # load model
    model = Glow(width=512, depth=32, n_levels=3, input_dims=(3, 64, 64))

    if args.load:
        model = torch.load(args.load)

    model = model.to(args.device)

    # load optimizers
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.trainable:
        image_folder = 'images'
        weight_folder = 'weights/nf'

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        os.makedirs(image_folder)

        if os.path.exists(weight_folder):
            shutil.rmtree(weight_folder)

        os.makedirs(weight_folder)

        train(model, train_loader, optimizer, args)


    # index2name = [
        # '5_o_Clock_Shadow',
        # 'Arched_Eyebrows',
        # 'Attractive',
        # 'Bags_Under_Eyes',
        # 'Bald',
        # 'Bangs',
        # 'Big_Lips',
        # 'Big_Nose',
        # 'Black_Hair',
        # 'Blond_Hair',
        # 'Blurry',
        # 'Brown_Hair',
        # 'Bushy_Eyebrows',
        # 'Chubby',
        # 'Double_Chin',
        # 'Eyeglasses',
        # 'Goatee',
        # 'Gray_Hair',
        # 'Heavy_Makeup',
        # 'High_Cheekbones',
        # 'Male',
        # 'Mouth_Slightly_Open',
        # 'Mustache',
        # 'Narrow_Eyes',
        # 'No_Beard',
        # 'Oval_Face',
        # 'Pale_Skin',
        # 'Pointy_Nose',
        # 'Receding_Hairline',
        # 'Rosy_Cheeks',
        # 'Sideburns',
        # 'Smiling',
        # 'Straight_Hair',
        # 'Wavy_Hair',
        # 'Wearing_Earrings',
        # 'Wearing_Hat',
        # 'Wearing_Lipstick',
        # 'Wearing_Necklace',
        # 'Wearing_Necktie',
        # 'Young'
    # ]

    # gen_labels = _gen_label(16)

    # for gen_label in gen_labels:
        # used_label = []

        # for i in range(len(gen_label)):
            # if gen_label[i] > 0.0:
                # used_label.append(index2name[i])

        # print(', '.join(used_label))

    # gen_labels = gen_labels.to(args.device)
    # samples = generate(model, gen_labels, n_samples=gen_labels.shape[0], z_stds=[1.0])

    # images = make_grid(samples.cpu(), nrow=4, normalize=True)
    # save_image(images, 'result.png')

    # model.eval()
    # for image, label in train_loader:
        # image = image.to(args.device)
        # label = label.to(args.device)

        # zs, _ = model.forward(image, label)

        # start_z = []
        # stop_z = []

        # for z in zs:
            # start_z.append(z[0:3])
            # stop_z.append(z[3:6])

        # output = torch.zeros(3, 200, 476)

        # samples, _ = model.inverse(label[0:3], start_z, batch_size=3, z_std=1.0)
        # images = make_grid(samples.cpu(), nrow=1, normalize=True)
        # print(image.shape)
        # output[:, :, 0: 68] = images

        # for i in range(5):
            # for j in range(len(start_z)):
                # start_z[j] += (stop_z[j] - start_z[j]) / 5.0

            # condition = label[0:3] if i < 5 else label[3:6]
            # samples, _ = model.inverse(condition, start_z, batch_size=3, z_std=1.0)
            # images = make_grid(samples.cpu(), nrow=1, normalize=True)
            # output[:, :, ((i + 1) * 68): ((i + 2) * 68)] = images

        # samples, _ = model.inverse(label[3:6], stop_z, batch_size=3, z_std=1.0)
        # images = make_grid(samples.cpu(), nrow=1, normalize=True)
        # output[:, :, (6 * 68):] = images

        # save_image(output, f'result.png')

        # break

    # for image, label in train_loader:
        # image = image.to(args.device)
        # label = label.to(args.device)

        # zs, _ = model.forward(image, label)

        # z = []
        # for element in zs:
            # z.append(element[0: 2])

        # label = torch.cat([label[0].reshape(1, 40), label[-1].reshape(1, 40)], dim=0)
        # print(label.shape)

        # for one_label in label:
            # used_label = []

            # for i in range(len(one_label)):
                # if one_label[i] > 0.0:
                    # used_label.append(index2name[i])

            # print(', '.join(used_label))

        # output = torch.zeros(3, 134, 68 * 5)

        # samples, _ = model.inverse(label, z, batch_size=label.shape[0], z_std=1.0)
        # images = make_grid(samples.cpu(), nrow=1, normalize=True)
        # print(image.shape)
        # output[:, :, 0: 68] = images

        # count = 0
        # while count < 4:
            # index = random.choice(range(40))
            # if label[0, index] < 0.0 and label[1, index] < 0.0:
                # count += 1

                # label[0, index] = 1.0
                # label[1, index] = 1.0

                # for one_label in label:
                    # used_label = []

                    # for i in range(len(one_label)):
                        # if one_label[i] > 0.0:
                            # used_label.append(index2name[i])

                    # print(', '.join(used_label))

                # samples, _ = model.inverse(label, z, batch_size=label.shape[0], z_std=1.0)
                # images = make_grid(samples.cpu(), nrow=1, normalize=True)
                # print(image.shape)
                # output[:, :, (count * 68): ((count + 1) * 68)] = images

        # save_image(output, f'result.png')

        # break
