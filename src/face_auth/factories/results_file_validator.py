"""Validator for results file handling."""

import os
from pathlib import Path
from face_auth.config.logging_config import get_logger

logger = get_logger(__name__)


class ResultsFileValidator:
    """Handles validation and user interaction for results file."""

    @staticmethod
    def validate(results_path: Path) -> None:
        """Validate results file, prompting user if file exists.

        Args:
            results_path: Path to results file

        Raises:
            RuntimeError: If user cancels or makes invalid choice
        """
        if not os.path.exists(results_path):
            return

        logger.warning(f"Results file already exists: {results_path}")
        print("  1. Delete the file and start fresh")
        print("  2. Append to the existing file")
        print("  3. Cancel execution")
        choice = input("\nChoice (1, 2, or 3): ").strip()

        if choice == '1':
            os.remove(results_path)
            logger.info("File deleted")
        elif choice == '2':
            logger.info("Will append to existing file")
        elif choice == '3':
            logger.info("Execution cancelled")
            raise RuntimeError("User cancelled execution")
        else:
            logger.error(f"Invalid choice: {choice}")
            raise RuntimeError("Invalid choice")