import torch
import torch.optim as optim
import cv2

from Utils.ContainerUtils import torch2numpy
from Utils.Loss import l1_loss, ssim_loss
from Utils.DataLoader import DataLoader
from GaussianModel import GaussianModel
from GaussianRenderer import GaussianRenderer


class Trainer:
    def __init__(self, model: GaussianModel, dataloader: DataLoader,
                 lr: float = 1e-3,
                 betas: tuple[float, float] = (0.5, 0.999),
                 train_steps: int = 2000,
                 refine_cycle: int = 500,
                 loss_lambda: float = 0.2,
                 result_dir: str = "result/",
                 debug: bool = False):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas)
        self.train_steps = train_steps
        self.refine_cycle = refine_cycle
        self.loss_lambda = loss_lambda
        self.result_dir = result_dir
        self.debug = debug

        self.renderer = GaussianRenderer(self.model, self.debug)

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ssim = ssim_loss(pred, gt)
        l1 = l1_loss(pred, gt)
        return (1 - self.loss_lambda) * l1 + self.loss_lambda * ssim

    def on_refinement_iteration(self):
        pass

    def get_image_scale(self, step: int):
        if step < 250:
            return 0.25
        elif step < 500:
            return 0.5
        return 1

    def on_train_step(self, step: int):
        cam, target_image = self.dataloader.sample(is_train=True, image_scale=self.get_image_scale(step))
        self.optimizer.zero_grad()

        image = self.renderer(cam)
        loss = self.compute_loss(image, target_image)
        loss.backward()
        self.optimizer.step()

    def train(self):
        train_step = 0
        while train_step < self.train_steps:
            self.on_train_step(train_step)
            if train_step > 0 and train_step % self.refine_cycle:
                self.on_refinement_iteration()
            train_step += 1

    def evaluate(self):
        cam, target_image = self.dataloader.sample(is_train=False)
        image = self.renderer.render(cam, tile_length=64)
        np_image = torch2numpy(image.detach())
        result = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite("result.jpg", result * 255)
        cv2.imwrite("target.jpg", cv2.cvtColor(torch2numpy(target_image), cv2.COLOR_RGB2BGR))
