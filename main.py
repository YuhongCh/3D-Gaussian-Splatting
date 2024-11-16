import yaml
import argparse

import cv2

from Utils.DataLoader import DataLoader
from GaussianModel import GaussianModel
from GaussianRenderer import GaussianRenderer
from Utils.ContainerUtils import torch2numpy


parser = argparse.ArgumentParser(description='3D Gaussian Splatting')
parser.add_argument('--config', default='configs/playroom.yaml')


def main():
    # Load the parameters
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)

    dataloader = DataLoader(image_dir="dataset/db/playroom/images", sfm_dir="sfm_directory")
    dataloader.extract_keypoint()
    pcd = dataloader.get_pcd()

    model = GaussianModel.from_pcd(pcd)
    renderer = GaussianRenderer(model)
    cam = dataloader.get_random_camera()
    result = renderer.render(cam).detach().cpu().numpy()
    result = cv2.cvtColor(result.transpose((1, 0, 2)), cv2.COLOR_RGB2BGR)
    cv2.imwrite("result.jpg", result * 255 * 3)
    cv2.imwrite("target.jpg", cv2.cvtColor(torch2numpy(cam.target_image), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()