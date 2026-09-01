from pathlib import Path

from torchgeo.datasets import BigEarthNetV2


DATA_ROOT = Path("data/bigearthnet_v2")


def get_dataset():
    return BigEarthNetV2(
        root=DATA_ROOT,
        split="test",
        bands="s2",
        download=False,
    )


if __name__ == "__main__":
    dataset = get_dataset()

    print("Dataset loaded")
    print("Number of samples:", len(dataset))