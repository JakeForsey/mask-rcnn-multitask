from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from model import multitask_maskrcnn_resnet50_fpn
from data import BuildingDataset
from plot import plot_from_results


def collate_fn(batch):
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return [images, targets]


def train_one_epoch(model, optimizer, data_loader):
    model.train()
    device = torch.device("cuda")
    loss_dicts = []
    for idx, (images, targets) in enumerate(data_loader):
        images = [image.to(device) for image in images]
        targets = [
            {k: v.to(device) for k, v in target.items() if k in [
                "masks", "boxes", "labels", "roof_style", "roof_material"
            ]}
            for target in targets
        ]

        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_dicts.append({k: v.detach() for k, v in loss_dict.items()})
        print(idx, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss_dicts


def test_one_epoch(model, data_loader, epoch):
    with torch.no_grad():
        model.eval()
        for idx, (images, target) in enumerate(data_loader):
            images = [image.cuda() for image in images]
            res = model(images)
            counter = 0

            def on_plot():
                nonlocal counter
                nonlocal idx
                results_dir = Path(f"results/{epoch}")
                results_dir.mkdir(parents=True, exist_ok=True)
                plt.savefig(results_dir / Path(f"{idx},{counter}"))
                counter += 1

            plot_from_results(res, images, on_plot)


def main():
    train_dataset = BuildingDataset(
        "data/ballarat.tif",
        "data/ballarat.geojson",
        fraction=0.3,
    )
    test_dataset = BuildingDataset(
        "data/keller.tif",
        "data/keller.geojson",
        bands=(1, 2, 3),
        fraction=0.1,
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=6,
        collate_fn=collate_fn
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=6,
        collate_fn=collate_fn
    )

    model = multitask_maskrcnn_resnet50_fpn()
    model = model.cuda()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )

    loss_dicts = []
    for epoch in range(100):
        print(f"TRAINING EPOCH: {epoch}")
        loss_dicts.append(
            train_one_epoch(model, optimizer, train_data_loader)
        )
        results_dir = Path(f"results/{epoch}")
        results_dir.mkdir(exist_ok=True, parents=True)
        dict0 = loss_dicts[0][0]
        for key in dict0:
            plt.plot(
                [sum([d[key] for d in dicts]) / len(dicts) for dicts in loss_dicts],
                label=key
            )
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / Path("losses.png"))
        print(f"SAVING CHECKPOINT: {epoch}")
        torch.save(model.state_dict(), f"checkpoints/{epoch}.pth")

        print(f"TESTING EPOCH: {epoch}")
        test_one_epoch(model, test_data_loader, epoch)


if __name__ == "__main__":
    main()
