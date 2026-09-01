from pprint import pprint

from app.vlm.p2_client import query_p2
from app.vlm.evidence_builder import build_evidence


QUERY = (
    "Show vegetation change from 2021 to 2025"
)


def main():

    print("=" * 70)
    print("REAL P2 → P4 EVIDENCE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Call the real Person 2 backend.
    # ---------------------------------------------------------

    print(
        "\nCalling P2 /api/query..."
    )

    p2_response = query_p2(
        QUERY
    )

    # ---------------------------------------------------------
    # 2. Build P4 evidence.
    # ---------------------------------------------------------

    evidence = build_evidence(
        p2_response
    )

    # ---------------------------------------------------------
    # 3. Show structured evidence.
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "P4 EVIDENCE"
    )

    print(
        "=" * 70
    )

    pprint(
        evidence
    )

    # ---------------------------------------------------------
    # 4. Explicit image information.
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "IMAGERY AVAILABLE TO P4"
    )

    print(
        "=" * 70
    )

    for image in evidence["images"]:

        print(
            "\nID:",
            image["id"],
        )

        print(
            "Date:",
            image["date"],
        )

        print(
            "Cloud cover:",
            image["cloud_cover"],
        )

        print(
            "Bands:"
        )

        for band, path in image[
            "bands"
        ].items():

            print(
                f"  {band}: {path}"
            )

    # ---------------------------------------------------------
    # 5. Visualization URLs.
    # ---------------------------------------------------------

    print(
        "\nVisualization URLs:"
    )

    for url in evidence[
        "visualizations"
    ]:

        print(
            " -",
            url,
        )


if __name__ == "__main__":
    main()