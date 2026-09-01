from PIL import Image

from app.vlm.model import VLM
from app.vlm.evidence_builder import build_evidence


def main():

    print("=" * 70)
    print("P2 → EVIDENCE → P4 VLM")
    print("=" * 70)

    # ---------------------------------------------------------
    # Simulated Person 2 API response.
    #
    # This matches the structure you showed us.
    # ---------------------------------------------------------

    p2_response = {
        "query": (
            "Show vegetation change "
            "from 2021 to 2025"
        ),

        "plan": {
            "task": "change_detection",
            "target": "vegetation",
            "metric": "ndvi",
            "time_start": 2021,
            "time_end": 2025,
        },

        "statistics": {
            "mean_before": 0.4201,
            "mean_after": 0.3091,
            "mean_change": -0.1110,
            "changed_pixels": 5,
            "valid_pixels": 9,
            "change_ratio": 0.5556,
            "change_type": "decrease",
            "threshold": 0.05,
        },

        "evidence": {
            "visualization_url":
                "/visualizations/"
                "vegetation_2021_2025_change.png"
        },

        "layers": {},

        "answer": (
            "NDVI decreased from "
            "0.4201 to 0.3091."
        ),
    }

    # ---------------------------------------------------------
    # Build the P4 evidence object.
    # ---------------------------------------------------------

    evidence = build_evidence(
        p2_response
    )

    print("\nEvidence built successfully.")

    # ---------------------------------------------------------
    # Load the sample image.
    #
    # This is only a test image for now.
    # Later we'll use the actual image associated
    # with the P2 analysis.
    # ---------------------------------------------------------

    image = Image.open(
        "data/bigearthnet/sample_rgb.png"
    )

    # ---------------------------------------------------------
    # Load VLM.
    # ---------------------------------------------------------

    print(
        "\nLoading VLM..."
    )

    vlm = VLM()

    # ---------------------------------------------------------
    # Ask VLM using image + P2 evidence.
    # ---------------------------------------------------------

    print(
        "Sending image + question + evidence..."
    )

    answer = vlm.generate(
        image=image,
        question=p2_response["query"],
        evidence=evidence,
    )

    # ---------------------------------------------------------
    # Display.
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print("\nQUESTION:")
    print(
        p2_response["query"]
    )

    print("\nP2 EVIDENCE:")
    print(evidence)

    print("\nVLM ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()