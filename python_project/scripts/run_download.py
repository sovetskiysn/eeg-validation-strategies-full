"""Download all source datasets into ``datasets/archive``."""

from dataset_download import download_distinguishing, download_sam40


def main() -> None:
    download_sam40()
    download_distinguishing()


if __name__ == "__main__":
    main()
