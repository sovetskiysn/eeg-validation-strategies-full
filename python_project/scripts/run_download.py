"""Download all source datasets into ``datasets/archive``."""

from dataset_download import (
    download_cognitive_tasks_eeg,
    download_distinguishing,
    download_gradcpt,
    download_sam40,
    download_selective_attention,
)


def main() -> None:
    download_sam40()
    download_distinguishing()
    download_gradcpt()
    download_selective_attention()
    download_cognitive_tasks_eeg()


if __name__ == "__main__":
    main()
