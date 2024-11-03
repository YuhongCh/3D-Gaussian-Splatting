import yaml
import argparse
from Utils.DataLoader import DataLoader
from Utils.PointCloud import PointCloud

from GaussianModel import GaussianModel
from GaussianRender import GaussianRender


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

    # load SfM data
    loader = DataLoader(args.input_path, args.output_path)
    sfm_scene = loader.load(args.output_index)
    pcd = PointCloud()
    pcd.set_coords()



if __name__ == "__main__":
    main()