from datasets import load_dataset


DATASET_NAME = "BIFOLD-BigEarthNetv2-0/BigEarthNet.txt"


def get_sample(index: int = 0):
    """
    Get one sample from BigEarthNet.txt.

    Uses streaming so the full 9.5M-row dataset
    is never loaded into memory.
    """

    dataset = load_dataset(
        DATASET_NAME,
        split="all_data",
        streaming=True,
    )

    for i, sample in enumerate(dataset):
        if i == index:
            return sample

    raise IndexError(
        f"Sample index {index} was not found."
    )


def get_sample_by_type(
    sample_type: str,
    start_index: int = 0,
    max_scan: int = 10000,
):
    """
    Find the first sample matching the requested
    BigEarthNet task type.
    """

    dataset = load_dataset(
        DATASET_NAME,
        split="all_data",
        streaming=True,
    )

    for i, sample in enumerate(dataset):

        if i < start_index:
            continue

        if sample["type"] == sample_type:
            return sample

        if i >= max_scan:
            break

    raise RuntimeError(
        f"Could not find task type '{sample_type}'."
    )


def print_sample(sample):
    """
    Print the important fields of one sample.
    """

    print("\nPATCH ID:")
    print(sample["patch_id"])

    print("\nS1 NAME:")
    print(sample["s1_name"])

    print("\nQUESTION:")
    print(sample["input"])

    print("\nGROUND TRUTH:")
    print(sample["output"])

    print("\nTYPE:")
    print(sample["type"])

    print("\nCATEGORY:")
    print(sample["category"])


if __name__ == "__main__":

    # For our current experiment, get a bounding-box task.
    sample = get_sample_by_type(
        "bounding box"
    )

    print_sample(sample)