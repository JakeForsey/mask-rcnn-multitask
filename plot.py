import json

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import box as shapely_box

from constants import ROOF_MATERIALS, ROOF_STYLES


def plot_from_results(results, images, on_plot=lambda: None):
    lookup = {
        "roof_style": ROOF_STYLES,
        "roof_material": ROOF_MATERIALS
    }
    for result, image in zip(results, images):
        try:
            result_str = json.dumps(
                {name: lookup[name][preds[0].argmax().item()] for name, preds in result['task_results'].items()},
                indent=True,
            )
            _plot(
                image.detach().cpu().numpy().transpose(1, 2, 0),
                result["boxes"][0].cpu().numpy(),
                result["masks"][0].cpu().numpy().squeeze(0),
                # Take the argmax of the first prediction for each task
                result_str,
            )
            on_plot()
            plt.clf()
        except IndexError as e:
            print("Unable to plot sample (probably no boxes predicted).")


def _plot(image, box, mask, title):
    plt.title(title, loc="left")
    plt.axis('off')
    plt.imshow(image)
    plt.plot(*shapely_box(*box).exterior.xy)
    plt.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.3)
    plt.tight_layout()
