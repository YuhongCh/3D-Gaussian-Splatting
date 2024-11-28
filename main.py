import os
import yaml
import argparse

from Utils.DataLoader import DataLoader
from GaussianModel import GaussianModel
from Trainer import Trainer


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

    dataloader = DataLoader(image_dir="dataset/db/playroom/images", sfm_dir="sfm_directory", seed=123)
    dataloader.extract_keypoint()
    dataloader.split_train_validate_data()
    dataloader.compute_scene_radius()

    model_path = "checkpoints/model0.ply"
    if not os.path.exists(model_path):
        pcd = dataloader.get_pcd()
        model = GaussianModel.from_pcd(pcd)
    else:
        model = GaussianModel()
        model.load_ply(model_path)


    trainer = Trainer(model, dataloader)
    trainer.evaluate()
    # trainer.train()
    # model.save_ply(model_path)


if __name__ == "__main__":
    main()