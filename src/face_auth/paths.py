"""Project-root path resolution.

Config data locations are written relative to the project root so they work
regardless of the directory the application is launched from. This module locates
that root and resolves relative paths against it, while leaving absolute paths
untouched.
"""
from pathlib import Path

# This file lives at <project_root>/src/face_auth/paths.py, so the project root
# is three levels up (paths.py -> face_auth -> src -> project_root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Fixed location for debug artifacts, anchored to the project root so it does not
# depend on the working directory the application was launched from.
DEBUG_DIR = PROJECT_ROOT / "src" / "debug"
NO_FACE_FRAMES_DIR = DEBUG_DIR / "no_face_frames"


def resolve_under_root(path: Path) -> Path:
    """Resolve a config path against the project root.

    Absolute paths are returned unchanged. Relative paths are anchored to the
    project root, so they resolve to the same location no matter which working
    directory the application was started from.

    Args:
        path: Path from configuration, absolute or relative to the project root

    Returns:
        Absolute path
    """
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
