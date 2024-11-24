import torch
import torch.optim as optim
import cv2

from Utils.ContainerUtils import torch2numpy, numpy2torch
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
        self.scene_radius = numpy2torch(self.dataloader.scene_radius, device=self.renderer.device).max()

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ssim = ssim_loss(pred, gt)
        l1 = l1_loss(pred, gt)
        return (1 - self.loss_lambda) * l1 + self.loss_lambda * ssim

    def get_image_scale(self, step: int):
        if step < 250:
            return 0.25
        elif step < 500:
            return 0.5
        return 1

    def train(self):
        for train_step in range(1, self.train_steps + 1):
            print(f"----------------- Iteration {train_step} -----------------")
            rendered_image = None
            while rendered_image is None:
                self.optimizer.zero_grad(set_to_none=True)
                cam, target_image = self.dataloader.sample(is_train=True, image_scale=self.get_image_scale(train_step))
                rendered_image, visible_mask, screen_coords = self.renderer(cam)
            loss = self.compute_loss(rendered_image, target_image)
            loss.backward()

            # try densify the model
            with torch.no_grad():
                self.model.update_densify_stats(screen_coords, visible_mask)
                if train_step > 0 and train_step % 500 == 0:
                    self.model.add_sh_degree()
                if train_step > 0 and train_step % 50 == 0:
                    self.model.densify(self.scene_radius)
                if train_step > 0 and train_step % 1500 == 0:
                    self.model.reset_opacity()
                if train_step > 0 and train_step % 500 == 0:
                    print(f"Train step {train_step}: Start evaluate and save checkpoint")
                    self.evaluate(train_step)
                    torch.save((self.model.capture(), train_step), f"checkpoints/checkpoint{train_step}.pth")

            self.optimizer.step()

    def evaluate(self, train_step: int):
        cam, target_image = self.dataloader.sample(is_train=False)
        image, _, _ = self.renderer.render(cam, tile_length=64)
        np_image = torch2numpy(image.detach())
        result = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        print(f"Validation Loss is {self.compute_loss(image, target_image)}")
        cv2.imwrite(f"Validate/result{train_step}.jpg", result * 255)
        cv2.imwrite(f"Validate/target{train_step}.jpg", cv2.cvtColor(torch2numpy(target_image), cv2.COLOR_RGB2BGR))
