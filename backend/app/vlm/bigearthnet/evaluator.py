import re


YES_VALUES = {
    "yes",
    "y",
    "true",
}

NO_VALUES = {
    "no",
    "n",
    "false",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.
    """

    text = str(text).strip().lower()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# BINARY
# ============================================================

def extract_binary_answer(text: str):
    """
    Extract the first clear yes/no answer.
    """

    normalized = normalize_text(text)

    for word in normalized.split():

        if word in YES_VALUES:
            return "yes"

        if word in NO_VALUES:
            return "no"

    return None


def evaluate_binary(
    prediction: str,
    reference: str,
) -> dict:
    """
    Evaluate BigEarthNet binary QA.
    """

    predicted = extract_binary_answer(
        prediction
    )

    expected = extract_binary_answer(
        reference
    )

    return {
        "prediction": prediction,
        "reference": reference,
        "normalized_prediction": predicted,
        "normalized_reference": expected,
        "correct": (
            predicted is not None
            and expected is not None
            and predicted == expected
        ),
    }


# ============================================================
# MCQ
# ============================================================

def extract_mcq_answer(text: str):
    """
    Extract an MCQ option letter.
    """

    normalized = normalize_text(text)

    match = re.search(
        r"\b([abcd])\b",
        normalized,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"(?:^|\s|\(|\[)([abcd])(?:\)|\.|:|$)",
        normalized,
    )

    if match:
        return match.group(1)

    return None


def evaluate_mcq(
    prediction: str,
    reference: str,
) -> dict:
    """
    Evaluate BigEarthNet MCQ.
    """

    predicted = extract_mcq_answer(
        prediction
    )

    expected = extract_mcq_answer(
        reference
    )

    return {
        "prediction": prediction,
        "reference": reference,
        "normalized_prediction": predicted,
        "normalized_reference": expected,
        "correct": (
            predicted is not None
            and expected is not None
            and predicted == expected
        ),
    }


# ============================================================
# BOUNDING BOX
# ============================================================

def extract_bbox(text: str):
    """
    Extract [x1, y1, x2, y2] from text.
    """

    if not text:
        return None

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        str(text),
    )

    if len(numbers) < 4:
        return None

    try:
        values = [
            float(numbers[i])
            for i in range(4)
        ]
    except ValueError:
        return None

    x1, y1, x2, y2 = values

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    return [
        left,
        top,
        right,
        bottom,
    ]


def bbox_iou(
    box_a,
    box_b,
) -> float:

    if box_a is None or box_b is None:
        return 0.0

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection_width = max(
        0.0,
        ix2 - ix1,
    )

    intersection_height = max(
        0.0,
        iy2 - iy1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_bbox(
    prediction: str,
    reference: str,
) -> dict:

    predicted_box = extract_bbox(
        prediction
    )

    reference_box = extract_bbox(
        reference
    )

    iou = bbox_iou(
        predicted_box,
        reference_box,
    )

    return {
        "prediction": prediction,
        "reference": reference,
        "predicted_bbox": predicted_box,
        "reference_bbox": reference_box,
        "iou": iou,
        "correct_iou_50": iou >= 0.50,
    }


# ============================================================
# CAPTIONING
# ============================================================

CAPTION_CONCEPTS = {
    "agriculture": [
        "agricultural",
        "agriculture",
        "arable",
        "crop",
        "crops",
        "farmland",
        "farm",
    ],
    "forest": [
        "forest",
        "forested",
        "woodland",
        "woods",
    ],
    "pasture": [
        "pasture",
        "pastures",
        "grassland",
        "grasslands",
    ],
    "mixed_landscape": [
        "diverse",
        "mixed",
        "mosaic",
        "landscape",
    ],
    "austria": [
        "austria",
        "austrian",
    ],
    "summer": [
        "summer",
    ],
}


def contains_concept(
    text: str,
    keywords,
) -> bool:

    normalized = normalize_text(text)

    return any(
        keyword in normalized
        for keyword in keywords
    )


def evaluate_caption(
    prediction: str,
    reference: str,
) -> dict:
    """
    Lightweight concept-coverage evaluation for
    BigEarthNet captions.

    This is NOT a replacement for a proper
    semantic caption metric. It is a transparent
    prototype metric for the current pipeline.
    """

    prediction_normalized = normalize_text(
        prediction
    )

    reference_normalized = normalize_text(
        reference
    )

    reference_concepts = []
    matched_concepts = []

    for concept, keywords in CAPTION_CONCEPTS.items():

        if contains_concept(
            reference_normalized,
            keywords,
        ):
            reference_concepts.append(
                concept
            )

            if contains_concept(
                prediction_normalized,
                keywords,
            ):
                matched_concepts.append(
                    concept
                )

    if reference_concepts:

        coverage = (
            len(matched_concepts)
            / len(reference_concepts)
        )

    else:

        coverage = 0.0

    return {
        "prediction": prediction,
        "reference": reference,
        "reference_concepts": reference_concepts,
        "matched_concepts": matched_concepts,
        "concept_coverage": coverage,
        "concept_coverage_percent": round(
            coverage * 100,
            2,
        ),
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EVALUATOR TEST")
    print("=" * 60)

    binary_result = evaluate_binary(
        prediction=(
            "Yes, it appears that arable land "
            "is adjacent to pastures."
        ),
        reference="yes",
    )

    print("\nBINARY:")
    print(binary_result)

    mcq_result = evaluate_mcq(
        prediction="c",
        reference="c",
    )

    print("\nMCQ:")
    print(mcq_result)

    bbox_result = evaluate_bbox(
        prediction="[0.0, 0.0, 0.7, 1.0]",
        reference="[0.0 0.0, 0.7 1.0]",
    )

    print("\nBBOX:")
    print(bbox_result)

    caption_result = evaluate_caption(
        prediction=(
            "This is a rural agricultural landscape "
            "with forested areas and pastures."
        ),
        reference=(
            "This satellite image, captured in Austria "
            "during summer, depicts agricultural and "
            "forested areas with pastures."
        ),
    )

    print("\nCAPTION:")
    print(caption_result)