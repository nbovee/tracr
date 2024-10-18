import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.experiment_design.datasets.imagenet import ImagenetDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_imagenet_subsets(root: Path, subset_sizes: list[int]):
    for n in subset_sizes:
        logger.info(f"Generating imagenet{n}_tr dataset")
        ImagenetDataset.imagenet_n_tr(root=root, n=n)
    logger.info("All subsets generated successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ImageNet subsets")
    parser.add_argument(
        "--root", type=str, required=True, help="Root directory of ImageNet dataset"
    )
    parser.add_argument(
        "--subsets",
        type=int,
        nargs="+",
        default=[2, 10, 50, 100],
        help="List of subset sizes to generate (default: [2, 10, 50, 100])",
    )

    args = parser.parse_args()
    root_path = Path(args.root)

    generate_imagenet_subsets(root_path, args.subsets)


# python scripts/generate_imagenet_subsets.py --root data/imagenet/ --subsets 2 10 50 100