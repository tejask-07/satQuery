from pathlib import Path

from PIL import Image


S1_DIR = Path(
    "data/s1_visualizations"
)


def main():
    print("=" * 70)
    print("S1 → P4 IMAGE TEST")
    print("=" * 70)

    images = {
        "s1_vv": S1_DIR / "s1_vv.png",
        "s1_vh": S1_DIR / "s1_vh.png",
        "s1_composite": S1_DIR / "s1_vv_vh_composite.png",
    }

    for name, path in images.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Missing image: {path}"
            )

        with Image.open(path) as img:
            print(
                f"{name}: "
                f"size={img.size}, "
                f"mode={img.mode}"
            )

    print(
        "\nAll S1 visualization images "
        "are ready for P4."
    )


if __name__ == "__main__":
    main()