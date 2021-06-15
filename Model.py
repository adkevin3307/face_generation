import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.functional import one_hot
from torchvision.utils import make_grid, save_image

from Net import Glow_Net


class BaseModel:
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self) -> None:
        raise NotImplementedError('train not implemented')

    def test(self) -> None:
        raise NotImplementedError('test not implemented')

    def load(self) -> None:
        raise NotImplementedError('load not implemented')

    def save(self) -> None:
        raise NotImplementedError('save not implemented')


class Glow(BaseModel):
    def __init__(self, net: Glow_Net, optimizer: optim.Optimizer) -> None:
        super(Glow, self).__init__()

        self.net = net
        self.optimizer = optimizer

        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _generate(self, condition: torch.Tensor, n_samples: int, z_stds: list) -> torch.Tensor:
        self.net.eval()

        samples = []
        for z_std in z_stds:
            sample, _ = self.net.inverse(None, condition, batch_size=n_samples, z_std=z_std)

            samples.append(sample)

        return torch.cat(samples, dim=0)

    def _gen_label(self, size: int, min_object_amount: int = 3, max_object_amount: int = 10, num_classes: int = 40) -> torch.Tensor:
        gen_label = []

        for _ in range(size):
            object_amount = np.random.randint(min_object_amount, max_object_amount, 1)

            temp_gen_label = np.random.choice(range(num_classes), object_amount, replace=False)
            temp_gen_label = one_hot(torch.tensor(temp_gen_label), num_classes)

            gen_label.append(torch.sum(temp_gen_label, dim=0).view(1, -1))

        return torch.cat(gen_label, dim=0).type(torch.float)

    def train(self, epochs: int, train_loader: DataLoader, warmup_epochs: int = 0, grad_norm_clip: float = 0.0) -> None:
        step = 0

        for epoch in range(epochs):
            self.net.train()

            tic = time.time()
            for i, (image, label) in enumerate(train_loader):
                image = image.to(self.device)
                label = label.to(self.device)

                step += 1

                # warmup learning rate
                if epoch <= warmup_epochs:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.optimizer.param_groups[0]['lr'] = lr * min(1, step / (len(train_loader) * warmup_epochs))

                self.optimizer.zero_grad()

                loss = -1.0 * torch.mean(self.net.log_prob(image, label, bits_per_pixel=True), dim=0)

                loss.backward()

                nn.utils.clip_grad_norm_(self.net.parameters(), grad_norm_clip)

                self.optimizer.step()

                et = time.time() - tic
                print(f'Epoch: [{epoch + 1}/{epochs}][{i + 1}/{len(train_loader)}], Time: {(et // 60):.0f}m{(et % 60):.02f}s, Loss: {loss.item():.3f}')

                if (i + 1) % 10 == 0:
                    samples = self._generate(label, n_samples=label.shape[0], z_stds=[1.0])

                    images = make_grid(samples.cpu(), nrow=4, pad_value=1, normalize=True)
                    save_image(images, f'images/generated_sample_{step}.png')

                    self.save(f'weights/nf/{epoch + 1}.pth')

    def test(self) -> None:
        raise RuntimeError('no test')

    def save(self, net_name: str) -> None:
        torch.save(self.net, net_name)

    def load(self, net_name: str) -> None:
        if net_name:
            self.net = torch.load(net_name)
            self.optimizer.param_groups[0]['params'] = self.net.parameters()

    @torch.no_grad()
    def generate(self, image_name: str, n_samples: int, z_stds: list) -> None:
        self.net.eval()

        condition = self._gen_label(n_samples)

        samples = []
        for z_std in z_stds:
            sample, _ = self.net.inverse(None, condition, batch_size=n_samples, z_std=z_std)

            samples.append(sample)

        samples = torch.cat(samples, dim=0)

        images = make_grid(samples.cpu(), nrow=int(n_samples ** 0.5), normalize=True)
        save_image(images, image_name)

    def interpolate(self, image_name: str, data_loader: DataLoader, n_samples: int = 3, interpolate_amount: int = 5) -> None:
        self.net.eval()

        image, label = next(iter(data_loader))
        image = image.to(self.device)
        label = label.to(self.device)

        zs, _ = self.net.forward(image, label)

        start_z = []
        stop_z = []

        n_samples = min(n_samples, label.shape[0])

        for z in zs:
            start_z.append(z[0: n_samples])
            stop_z.append(z[n_samples: (2 * n_samples)])

        output = []

        samples, _ = self.net.inverse(start_z, label[0: n_samples], batch_size=n_samples, z_std=1.0)
        images = make_grid(samples.cpu(), nrow=1, normalize=True)
        output.append(images)

        for i in range(interpolate_amount):
            z = []

            for j in range(len(start_z)):
                z[j].append((stop_z[j] - start_z[j]) / interpolate_amount * (i + 1))

            condition = label[0: n_samples] if i < (interpolate_amount // 2) else label[n_samples: (2 * n_samples)]

            samples, _ = self.net.inverse(z, condition, batch_size=n_samples, z_std=1.0)
            images = make_grid(samples.cpu(), nrow=1, normalize=True)
            output.append(images)

        samples, _ = self.net.inverse(stop_z, label[n_samples: (2 * n_samples)], batch_size=n_samples, z_std=1.0)
        images = make_grid(samples.cpu(), nrow=1, normalize=True)
        output.append(images)

        output = torch.cat(output, dim=-1)

        save_image(output, image_name)

    def change_attribute(self, image_name: str, data_loader: DataLoader, n_samples: int = 2, change_amount: int = 4) -> None:
        image, label = next(iter(data_loader))

        image = image.to(self.device)
        label = label.to(self.device)

        zs, _ = self.net.forward(image, label)

        n_samples = min(n_samples, label.shape[0])

        z = []
        for element in zs:
            z.append(element[0: n_samples])

        label = torch.cat([label[0, :], label[-1, :]], dim=0)

        output = []

        samples, _ = self.net.inverse(z, label, batch_size=label.shape[0], z_std=1.0)
        images = make_grid(samples.cpu(), nrow=1, normalize=True)
        output.append(images)

        count = 0
        while count < change_amount:
            index = random.choice(range(40))

            if label[0, index] < 0.0 and label[1, index] < 0.0:
                count += 1

                label[0, index] = 1.0
                label[1, index] = 1.0

                samples, _ = self.net.inverse(z, label, batch_size=label.shape[0], z_std=1.0)
                images = make_grid(samples.cpu(), nrow=1, normalize=True)
                output.append(images)

        output = torch.cat(output, dim=-1)

        save_image(output, image_name)
