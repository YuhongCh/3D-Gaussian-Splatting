from pathlib import Path
import pycolmap


class DataLoader:
    def __init__(self, image_dir: str, output_dir):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise OSError(f"{image_dir} Cannot be found")

        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir()

        self.database_path = self.output_dir / "database.db"

    def load(self, output_index: int) -> pycolmap.Reconstruction:
        sfm_dir = self.output_dir / str(output_index)
        if not sfm_dir.exists():
            print('[SfM] Failed to find the database, construct new one instead')
            pycolmap.extract_features(self.database_path, self.image_dir)
            pycolmap.match_exhaustive(self.database_path)
            maps = pycolmap.incremental_mapping(self.database_path, self.image_dir, self.output_dir)
            maps[output_index].write(self.output_dir)
        else:
            print('[SfM] Find the database, continue loading')
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(sfm_dir)
        reconstruction.check()
        print('[SfM] Database loading completed')
        return reconstruction


