import os
import sys
import logging
from face_auth.io import ConfigManager
from face_auth.utils import setup_logging, get_logger
from face_auth.models import ParticipantInfo
from face_auth.processing import process_participant


def initialize_logging(config_file):
    """Initialize logging based on the log level specified in the config file."""
    config = ConfigManager(config_file)
    log_level_str = config.get("log_level", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    setup_logging(log_level)

    return get_logger(__name__)


def validate_results_file(results_csv_path: str, logger) -> bool:
    """Check if results file exists and prompt user for deletion. Returns True to continue, False to stop."""
    if not os.path.exists(results_csv_path):
        return True

    logger.warning(f"Results file already exists at: {results_csv_path}")
    confirm = input("Do you want to delete this file and continue? (y/N): ").strip().lower()
    if confirm == 'y':
        os.remove(results_csv_path)
        logger.info("File deleted")
        return True
    else:
        logger.info("File NOT deleted. Stopping execution")
        return False


def main(config_file):
    logger = get_logger(__name__)

    config = ConfigManager(config_file)
    base_path = config.get("base_path")
    enrollment_base_path = config.get("enrollment_base_path")
    results_csv_path = config.get("results_file").format(base_path=base_path)

    participants = config.get("participants")
    devices = config.get("devices")
    pool_name = config.get("pool")

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing Pool: {pool_name.upper()}")
    logger.info(f"Base Path: {base_path}")
    logger.info(f"Enrollment Path: {enrollment_base_path}")
    logger.info(f"Devices: {', '.join(devices)}")
    logger.info(f"{'=' * 60}")

    if not validate_results_file(results_csv_path, logger):
        return

    for device in devices:
        for participant in participants:
            participant_info = ParticipantInfo(name=participant["name"], device=device)
            process_participant(
                participant=participant_info,
                base_path=base_path,
                enrollment_base_path=enrollment_base_path,
                results_csv_path=results_csv_path,
                config=config,
                logger=logger
            )

    logger.info(f"{'=' * 60}")
    logger.info(f"Processing complete! Results saved to: {results_csv_path}")
    logger.info(f"{'=' * 60}")


if __name__ == "__main__":
    # Allow config file to be passed as command line argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        print("No config file specified. Available configs:")
        print("  1. controlled_study_config.json")
        print("  2. in_the_wild_config.json")
        choice = input("\nSelect config (1 or 2): ").strip()

        if choice == "1":
            config_file = "controlled_study_config.json"
        elif choice == "2":
            config_file = "in_the_wild_config.json"
        else:
            print("Invalid choice. Using controlled_study_config.json")
            config_file = "controlled_study_config.json"

    startup_logger = initialize_logging(config_file)

    main(config_file)
