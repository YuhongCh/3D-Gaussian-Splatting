import torch
import torch.optim as optim
import cv2
from tqdm import tqdm

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
        self.loss_value = []

        self.renderer = GaussianRenderer(self.model, self.debug)
        self.scene_radius = numpy2torch(self.dataloader.scene_radius, device=self.renderer.device).max()

    def compute_loss(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        ssim = ssim_loss(pred, gt)
        l1 = l1_loss(pred, gt)
        return (1 - self.loss_lambda) * l1 + self.loss_lambda * (1 - ssim)

    def get_image_scale(self, step: int):
        if step < 250:
            return 0.25
        elif step < 500:
            return 0.5
        return 1

    def train(self):
        self.loss_value = []
        progress_bar = tqdm(range(0, self.train_steps), desc="Training progress")
        for train_step in range(1, self.train_steps + 1):
            rendered_image = None
            while rendered_image is None or not rendered_image.requires_grad:
                self.optimizer.zero_grad(set_to_none=True)
                cam, target_image = self.dataloader.sample(is_train=True)
                rendered_image, visible_mask, screen_coords, radius = self.renderer(cam)
            loss = self.compute_loss(rendered_image, target_image)
            loss.backward()

            # try densify the model
            with torch.no_grad():
                self.model.update_densify_stats(screen_coords, visible_mask)
                if train_step > 0 and train_step % 500 == 0:
                    self.model.add_sh_degree()
                if train_step > 0 and train_step % 50 == 0:
                    self.model.densify(self.scene_radius)

                    size_threshold = 20 if train_step > 1500 else 1000
                    remove_mask = (self.model.opacity < 0.005) & (radius > size_threshold)
                    self.model.remove(remove_mask)
                    torch.cuda.empty_cache()
                    print(f"There are {self.model._size} in size and {self.model.capacity} in capacity")

                if train_step > 0 and train_step % 1500 == 0:
                    self.model.reset_opacity()
                if train_step > 0 and train_step % 500 == 0:
                    print(f"Train step {train_step}: Start evaluate and save checkpoint")
                    self.evaluate(train_step)
                    torch.save((self.model.capture(self.optimizer), train_step), f"checkpoints/checkpoint{train_step}.pth")

            self.optimizer.step()
            if train_step % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss}"})
                progress_bar.update(10)
                self.loss_value.append(loss.item())
        progress_bar.close()

        with open('lose.txt', 'w+') as file:
            for loss in self.loss_value:
                file.write(loss + '\n')

    @torch.no_grad
    def evaluate(self, train_step: int = 0):
        cam, target_image = self.dataloader.sample(is_train=False)
        image, _, _, _ = self.renderer(cam)

        if image is None:
            print("Failed to evaluate")
            return

        np_image = torch2numpy(image.detach())
        result = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        print(f"Validation Loss is {self.compute_loss(image, target_image)}")
        cv2.imwrite(f"Validate/result{train_step}.jpg", result * 255)
        cv2.imwrite(f"Validate/target{train_step}.jpg", cv2.cvtColor(torch2numpy(target_image), cv2.COLOR_RGB2BGR))

    @torch.no_grad
    def test(self):
        cam, target_image = self.dataloader.sample(is_train=False)
        image, _, _, _ = self.renderer(cam)
        np_image = torch2numpy(image.detach()) * 255
        np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)

        screen_coords = torch.zeros_like(self.model.coords, dtype=torch.float32, device="cuda")
        pos2d, cov2d, mask = cam.project_gaussian(screen_coords, self.model.coords, self.model.covariance)
        np_pos2d = torch2numpy(pos2d.detach()).astype(int)
        for i in range(np_pos2d.shape[0]):
            image = cv2.circle(np_image, np_pos2d[i, :2], 1, (255, 0, 0))
        cv2.imwrite(f"test_result.jpg", image)
        cv2.imwrite(f"test_target.jpg", cv2.cvtColor(torch2numpy(target_image), cv2.COLOR_RGB2BGR))
