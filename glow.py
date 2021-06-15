"""
Glow: Generative Flow with Invertible 1x1 Convolutions
arXiv:1807.03039v2
"""

import os
import shutil
import random
import numpy as np
import torch
import torch.optim as optim
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

from utils import parse, load_data
from Net import Glow_Net
from Model import Glow


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
    net = Glow_Net(hidden_channels=512, depth=32, n_levels=3, input_dims=(3, 64, 64))

    if args.load:
        model = torch.load(args.load)

    # load optimizers
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    model = Glow(net, optimizer)

    if args.trainable:
        image_folder = 'images'
        weight_folder = 'weights/nf'

        if os.path.exists(image_folder):
            shutil.rmtree(image_folder)

        os.makedirs(image_folder)

        if os.path.exists(weight_folder):
            shutil.rmtree(weight_folder)

        os.makedirs(weight_folder)

        model.train(args.epochs, train_loader, warmup_epochs=args.warmup_epochs, grad_norm_clip=args.grad_norm_clip)
