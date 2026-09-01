from pathlib import Path

from PIL import Image

from app.vlm.model import VLM


IMAGE_PATH = Path(
    "data/bigearthnet/sample_rgb.png"
)

QUESTION = (
    "Would you say that any arable land lies next "
    "to pastures in the image? "
    "Answer only yes or no."
)


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    print("=" * 60)
    print("BIGEARTHNET VLM TEST")
    print("=" * 60)

    print("\nQuestion:")
    print(QUESTION)

    print("\nLoading VLM...")
    vlm = VLM()

    print("Sending image + question to VLM...")

    answer = vlm.generate(
        image=image,
        question=QUESTION,
    )

    print("\nVLM ANSWER:")
    print(answer)

    print("\nGROUND TRUTH:")
    print("yes")


if __name__ == "__main__":
    main()