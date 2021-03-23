import argparse
import os
import time

import cv2
import piq
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Adam
from torch.utils.data import DataLoader

from honorcup.data import SRDataset
from honorcup.network import RRDBNet


def init_weights(network: nn.Module) -> None:
    def init_func(module: nn.Module) -> None:
        if hasattr(module, 'weight'):
            init.normal_(module.weight.data, 0.0, 0.01)
            if hasattr(module, 'bias') and module.bias is not None:
                init.constant_(module.bias.data, 0.0)

    network.apply(init_func)


def run_train(
        network: nn.Module,
        criterion_l1: nn.Module,
        criterion_vgg: nn.Module,
        optimizer: Adam,
        train_dataloader: DataLoader,
        device: str) -> None:
    iter_id = 0
    start_time = time.time()
    for epoch_id in range(5):
        for lr_images, hr_images in train_dataloader:
            lr_images = lr_images.to(device)
            hr_images = hr_images.to(device)
            predictions = network(lr_images)
            loss_l1 = criterion_l1(predictions, hr_images)
            loss_vgg = 1e-3 * criterion_vgg(predictions, hr_images)
            loss = loss_l1 + loss_vgg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter_id % 100 == 0:
                print(
                    "iter_id", iter_id,
                    "time", time.time() - start_time,
                    "loss l1", loss_l1.item(),
                    "loss vgg", loss_vgg.item())
            iter_id += 1

        optimizer.param_groups[0]["lr"] /= 2


def run_test(
        network: nn.Module,
        test_dataloader: DataLoader,
        device: str,
        result_dir: str) -> None:
    metric = piq.DISTS().to(device)
    os.makedirs(result_dir, exist_ok=True)
    metric_values = []
    for i, (lr_images, hr_images) in enumerate(test_dataloader):
        lr_images = lr_images.to(device)
        hr_images = hr_images.to(device)
        with torch.no_grad():
            predictions = network(lr_images)
        metric_values.append(metric(predictions, hr_images))

        lr_image = (lr_images.cpu()[0].clamp(0, 1).permute(1, 2, 0) * 255).numpy().astype("uint8")
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, f"{i}_lr.png"), lr_image)

        hr_image = (hr_images.cpu()[0].clamp(0, 1).permute(1, 2, 0) * 255).numpy().astype("uint8")
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, f"{i}_hr.png"), hr_image)

        pred_image = (predictions.cpu()[0].clamp(0, 1).permute(1, 2, 0) * 255).numpy().astype("uint8")
        pred_image = cv2.cvtColor(pred_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(result_dir, f"{i}_pred.png"), pred_image)

    mean_metric = torch.stack(metric_values).mean()
    print(f"DISTS metric {mean_metric.item()}")

    torch.save(network.state_dict(), os.path.join(result_dir, "state.pth"))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("train_hr_dir")
    parser.add_argument("train_lr_dir")

    parser.add_argument("test_hr_dir")
    parser.add_argument("test_lr_dir")

    parser.add_argument("--result_dir", default="results")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = SRDataset(args.train_hr_dir, args.train_lr_dir, crop_size=64, length=40000)
    test_dataset = SRDataset(args.test_hr_dir, args.test_lr_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    network = RRDBNet().to(device)
    init_weights(network)
    optimizer = Adam(network.parameters(), lr=1e-4)
    criterion_vgg = piq.ContentLoss().to(device)
    criterion_l1 = nn.L1Loss()

    run_train(network, criterion_l1, criterion_vgg, optimizer, train_dataloader, device)
    run_test(network, test_dataloader, device, args.result_dir)


if __name__ == "__main__":
    main()
