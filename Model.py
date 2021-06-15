import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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

    def train(self, epochs: int, train_loader: DataLoader, valid_loader: DataLoader = None, warmup_epochs: int = 0, grad_norm_clip: float = 0.0) -> None:
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

            if valid_loader:
                self.test(valid_loader)

    def test(self, test_loader: DataLoader) -> None:
        pass

    def save(self, net_name: str) -> None:
        torch.save(self.net, net_name)

    def load(self, net_name: str) -> None:
        if net_name:
            self.net = torch.load(net_name)
            self.optimizer.param_groups[0]['params'] = self.net.parameters()
