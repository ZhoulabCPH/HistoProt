from __future__ import annotations

import argparse
import logging
from pathlib import Path

from histoprot.config import build_train_config, load_yaml_config, save_resolved_config, validate_config
from histoprot.engine import train_nested_cross_validation
from histoprot.runtime import configure_logging


LOGGER = logging.getLogger(__name__)


def parse_cli_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HistoProt with a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).with_name("config.yaml")),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging regardless of the YAML setting.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_cli_arguments()
    config = build_train_config(load_yaml_config(cli_args.config), cli_verbose=cli_args.verbose)
    validate_config(config)
    configure_logging(verbose=config.verbose)

    run_dir = Path(config.checkpoint_dir).resolve() / config.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, run_dir)

    LOGGER.info("HistoProt training started.")
    train_nested_cross_validation(config, run_dir)


if __name__ == "__main__":
    main()
