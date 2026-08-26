"""Convert both source datasets from ``datasets/archive`` into raw BIDS."""

from dataset_standardization import standardize_distinguishing, standardize_sam40


def main() -> None:
    standardize_sam40()
    standardize_distinguishing()


if __name__ == "__main__":
    main()
