from app.vlm.p2_client import query_p2
from app.vlm.evidence_builder import build_evidence
from app.vlm.p2_imagery import load_p2_images
from app.vlm.model import VLM


QUERY = "Show vegetation change from 2021 to 2025"


def main():

    print("=" * 70)
    print("REAL P2 → IMAGES + EVIDENCE → P4 VLM")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Real P2 API
    # ---------------------------------------------------------

    print(
        "\nCalling P2 /api/query..."
    )

    p2_response = query_p2(
        QUERY
    )

    # ---------------------------------------------------------
    # 2. Evidence
    # ---------------------------------------------------------

    evidence = build_evidence(
        p2_response
    )

    # ---------------------------------------------------------
    # 3. Load actual P2 imagery
    # ---------------------------------------------------------

    print(
        "\nLoading P2 imagery..."
    )

    images = load_p2_images(
        evidence
    )

    print(
        "Images available:",
        list(images.keys())
    )

    if "before" not in images:
        raise RuntimeError(
            "P2 before image was not loaded."
        )

    if "after" not in images:
        raise RuntimeError(
            "P2 after image was not loaded."
        )

    # ---------------------------------------------------------
    # 4. VLM
    # ---------------------------------------------------------

    print(
        "\nLoading VLM..."
    )

    vlm = VLM()

    # ---------------------------------------------------------
    # 5. Multimodal inference
    # ---------------------------------------------------------

    print(
        "Sending before image + after image "
        "+ evidence + question..."
    )

    answer = vlm.generate(
        question=QUERY,
        evidence=evidence,
        images=images,
    )

    # ---------------------------------------------------------
    # 6. Result
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        "\nQuestion:"
    )

    print(
        QUERY
    )

    print(
        "\nP2 statistics:"
    )

    print(
        evidence["statistics"]
    )

    print(
        "\nImages:"
    )

    for name, image in images.items():

        print(
            f"{name}:",
            image.size,
            image.mode,
        )

    print(
        "\nVLM ANSWER:"
    )

    print(
        answer
    )


if __name__ == "__main__":
    main()