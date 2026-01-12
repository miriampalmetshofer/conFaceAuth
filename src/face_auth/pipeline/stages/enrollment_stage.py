from face_auth.config.logging_config import get_logger
from face_auth.config.models import ProcessingContext
from face_auth.services.enrollment_service import EnrollmentService
from face_auth.services.models import EnrollmentData

logger = get_logger(__name__)

class EnrollmentStage:
    """Stage 2: Ensure enrollment exists or create it."""

    def __init__(self, enrollment_service: EnrollmentService):
        """Initialize with enrollment service.

        Args:
            enrollment_service: Service for managing enrollments
        """
        self.enrollment_service = enrollment_service

    def execute(self, context: ProcessingContext) -> EnrollmentData:
        """Setup enrollment for participant.

        Args:
            context: Processing context with participant and device

        Returns:
            Enrollment data with embeddings
        """
        logger.debug(f"Setting up enrollment for {context.participant.name}")

        enrollment_data = self.enrollment_service.get_enrollment(context)

        logger.debug("Enrollment ready")
        return enrollment_data