"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import os
import time
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image, make_grid

from utils import parse, load_data
from Net import Glow


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
    model = Glow(width=512, depth=32, n_levels=3, input_dims=(3, 64, 64)).to(args.device)

    if args.load:
        model = torch.load(args.load)

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
