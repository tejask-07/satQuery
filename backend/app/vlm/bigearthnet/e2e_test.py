from app.vlm.model import VLM

from app.vlm.bigearthnet.dataset import (
    get_sample,
)

from app.vlm.bigearthnet.remote_tar import (
    load_patch_bands,
    cache_info,
)

from app.vlm.bigearthnet.rgb import (
    create_rgb_image,
)

from app.vlm.bigearthnet.evaluator import (
    evaluate_binary,
    evaluate_mcq,
)


def normalize_task_type(task_type: str) -> str:
    """
    Normalize BigEarthNet task labels.
    """

    return str(task_type).strip().lower()


def evaluate_result(
    task_type: str,
    prediction: str,
    ground_truth: str,
):
    """
    Run the evaluator supported by the current prototype.
    """

    task_type = normalize_task_type(
        task_type
    )

    if task_type == "binary":

        return evaluate_binary(
            prediction=prediction,
            reference=ground_truth,
        )

    if task_type == "mcq":

        return evaluate_mcq(
            prediction=prediction,
            reference=ground_truth,
        )

    return None


def main():

    print("=" * 70)
    print("BIGEARTHNET MULTI-SAMPLE CACHE TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load the VLM once.
    # ---------------------------------------------------------

    print("\nLoading VLM...")

    vlm = VLM()

    # ---------------------------------------------------------
    # Process several consecutive annotations.
    # ---------------------------------------------------------

    NUM_SAMPLES = 5

    for index in range(NUM_SAMPLES):

        print("\n")
        print("#" * 70)
        print(
            f"SAMPLE {index + 1}/{NUM_SAMPLES}"
        )
        print("#" * 70)

        # -----------------------------------------------------
        # 1. Get BigEarthNet annotation.
        # -----------------------------------------------------

        sample = get_sample(
            index=index
        )

        patch_id = sample["patch_id"]
        question = sample["input"]
        ground_truth = sample["output"]
        task_type = sample["type"]

        print(
            "\nPatch:",
            patch_id,
        )

        print(
            "\nTask:",
            task_type,
        )

        print(
            "\nQuestion:",
            question,
        )

        print(
            "\nGround truth:",
            ground_truth,
        )

        # -----------------------------------------------------
        # 2. Load S2 patch.
        #
        # If another annotation uses the same patch,
        # load_patch_bands() will use the RAM cache.
        # -----------------------------------------------------

        print(
            "\nLoading S2 patch..."
        )

        bands = load_patch_bands(
            patch_id
        )

        # -----------------------------------------------------
        # 3. Convert S2 to RGB.
        # -----------------------------------------------------

        image = create_rgb_image(
            bands
        )

        print(
            "\nRGB:",
            image.size
        )

        # -----------------------------------------------------
        # 4. VLM inference.
        # -----------------------------------------------------

        print(
            "\nRunning VLM..."
        )

        prediction = vlm.generate(
            image=image,
            question=question,
        )

        print(
            "\nPrediction:",
            prediction,
        )

        # -----------------------------------------------------
        # 5. Evaluation.
        # -----------------------------------------------------

        result = evaluate_result(
            task_type=task_type,
            prediction=prediction,
            ground_truth=ground_truth,
        )

        if result is not None:

            print(
                "\nNormalized prediction:",
                result[
                    "normalized_prediction"
                ],
            )

            print(
                "Normalized reference:",
                result[
                    "normalized_reference"
                ],
            )

            print(
                "Correct:",
                result["correct"],
            )

        else:

            print(
                "\nEvaluator not implemented for:",
                task_type,
            )

    # ---------------------------------------------------------
    # Cache summary.
    # ---------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL CACHE STATE")
    print("=" * 70)

    cache_info()


if __name__ == "__main__":
    main()