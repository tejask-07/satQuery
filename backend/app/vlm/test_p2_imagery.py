from app.vlm.p2_client import query_p2
from app.vlm.evidence_builder import build_evidence
from app.vlm.p2_imagery import load_p2_images


def main():

    query = (
        "Show vegetation change "
        "from 2021 to 2025"
    )

    print("=" * 70)
    print("P2 IMAGERY TEST")
    print("=" * 70)

    p2_response = query_p2(
        query
    )

    evidence = build_evidence(
        p2_response
    )

    images = load_p2_images(
        evidence
    )

    print(
        "\nImages loaded:",
        list(images.keys())
    )

    for name, image in images.items():

        print(
            f"{name}:",
            image.size,
            image.mode,
        )

        output_path = (
            f"data/{name}_p2.png"
        )

        image.save(
            output_path
        )

        print(
            "Saved:",
            output_path
        )


if __name__ == "__main__":
    main()