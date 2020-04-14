import sys

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

COLUMNS = {
    "height": np.float64,
    "roof_style": np.uint32,
    "roof_material": np.uint32,
    "geometry": Polygon,
}


def process_dropbox_data(in_path, out_path):
    # Mapping from
    # https://gitlab.com/gsi-buildings/gsi-building-roof-style/-/blob/develop/config.train.yaml
    roof_style_mapping = {
        1: 0,
        2: 1,
        3: 2,
        4: 1,
        5: 0,
        6: 0,
        7: 3,
        8: 2,
        9: 3,
        10: 3,
        11: 3,
    }
    gdf = gpd.read_file(in_path)
    gdf["height"] = gdf["height_m"].astype(COLUMNS["height"])
    gdf["roof_style"] = gdf["roof_type1"].apply(lambda x: roof_style_mapping[x]).astype(COLUMNS["roof_style"])
    gdf["roof_material"] = gdf["roof_mat"].astype(COLUMNS["roof_material"])
    gdf = gdf.loc[:, COLUMNS.keys()]
    gdf.to_file(out_path, driver="GeoJSON")


def main():
    process_dropbox_data(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
