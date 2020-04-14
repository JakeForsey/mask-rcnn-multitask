from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import rasterio
from rasterio import windows
from rasterio.features import rasterize
from shapely.geometry import box as shapely_box
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from plot import _plot


def per_band(image, func, *args, **kwargs):
    normalized = np.empty(image.shape)
    for channel_index in range(image.shape[2]):
        normalized[:, :, channel_index] = func(image[:, :, channel_index], *args, **kwargs)
    return normalized


def quantile(image, q):
    min_ = np.nanquantile(image, q[0], interpolation='higher')
    max_ = np.nanquantile(image, q[1], interpolation='lower')
    image = (image - min_) / (max_ - min_)
    return np.clip(image, 0.0, 1.0)


class BuildingDataset(Dataset):
    def __init__(self, imagery_path, building_path, bands=(5, 3, 2), fraction=1.0):
        super().__init__()
        self._raster = rasterio.open(imagery_path)
        self._bands = bands
        gdf = gpd.read_file(building_path).sample(frac=fraction)
        # Filter geometries not fully within the raster extent
        raster_extent = shapely_box(*self._raster.bounds)
        self._gdf = gdf.loc[gdf.geometry.buffer(0.0005).within(raster_extent)].reset_index()
        self._image_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        self._cache_dir = Path("data/cache/")

    def __getitem__(self, index):
        image = self.image(index)
        image = self._image_transforms(image)
        target = self.target(index)
        return image, target

    def __len__(self):
        return len(self._gdf)

    def geom(self, index):
        return self._gdf.geometry[index]

    def extent_and_window(self, geom):
        image_extent = geom.centroid.buffer(0.0005).bounds
        window = windows.from_bounds(
            *image_extent,
            transform=self._raster.transform
        )
        return image_extent, window

    def target(self, index):
        geom = self.geom(index)
        image_extent, window = self.extent_and_window(geom)

        rounded = window.round_shape()
        mask = rasterize(
            ((g, 1) for g in [geom]),
            out_shape=(rounded.height, rounded.width),
            transform=self._raster.window_transform(window)
        )

        def box():
            pos = np.where(mask == 1)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            return [xmin, ymin, xmax, ymax]

        return {
            "boxes": torch.tensor([box()], dtype=torch.float32),
            # We have 2 classes, 0=background, 1=building.
            "labels": torch.tensor([1], dtype=torch.int64),
            "masks": torch.tensor([mask], dtype=torch.uint8),
            "roof_style": torch.tensor(
                self._gdf.loc[index, ["roof_style"]].values.astype(np.int), dtype=torch.int64
            ),
            "roof_material": torch.tensor(
                self._gdf.loc[index, ["roof_material"]].values.astype(np.int), dtype=torch.int64
            ),
        }

    def image(self, index):
        geom = self.geom(index)
        image_extent, window = self.extent_and_window(geom)

        image_path = self._cache_dir / Path(f"{window}.png")
        if not image_path.is_file():
            array = self._raster.read(
                # RGB
                indexes=self._bands,
                window=window.round_lengths()
            )
            array = array.transpose(1, 2, 0)
            array = per_band(array, quantile, q=(0.00, 0.99))
            array = array * 255
            array = array.astype(np.uint8)
            image = Image.fromarray(array, mode="RGB")
            image.save(image_path)
        else:
            image = Image.open(image_path)

        return image

    def plot(self, index):
        image = self.image(index)
        target = self.target(index)
        _plot(
            image,
            target["boxes"][0].numpy(),
            target["masks"][0].numpy(),
            f"roof_style: {target['roof_style']}, roof_material: {target['roof_material']}"
        )
        plt.show()


if __name__ == "__main__":
    ds = BuildingDataset(
        "data/keller.tif",
        "data/keller.geojson",
        bands=(1, 2, 3),
    )
    for i in range(5):
        ds.plot(i)

    ds = BuildingDataset(
        "data/ballarat.tif",
        "data/ballarat.geojson",
    )
    for i in range(5):
        ds.plot(i)
