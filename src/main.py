"""Face authentication system entry point."""

import sys
from face_auth.config import ConfigLoader
from face_auth.application import FaceAuthApplication
from face_auth.utils import setup_logging, get_logger


def parse_cli_arguments() -> str:
    """Parse command line arguments and return config file path."""
    if len(sys.argv) > 1:
        return sys.argv[1]

    # Interactive selection
    print("Select configuration:")
    print("  1. Controlled Study")
    print("  2. In The Wild")
    choice = input("\nChoice (1 or 2): ").strip()

    configs = {
        "1": "configs/controlled_study.json",
        "2": "configs/in_the_wild.json"
    }

    if choice not in configs:
        print(f"Invalid choice: {choice}")
        sys.exit(1)

    return configs[choice]


def main():
    """Application entry point."""
    # Parse CLI arguments
    config_file = parse_cli_arguments()

    # Load and validate configuration
    config_loader = ConfigLoader()
    config = config_loader.load(config_file)

    # Setup logging
    setup_logging(config.logging.get_log_level())
    logger = get_logger(__name__)

    logger.info(f"Loaded configuration from: {config_file}")

    try:
        app = FaceAuthApplication(config)
        app.run()

    except Exception as e:
        logger.exception(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
