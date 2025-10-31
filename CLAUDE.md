# Claude Code Development Instructions

## Docstrings
- Use triple double quotes (`"""`) for all docstrings.
- Mostly a oneliner docstring is preferred.
- No input/output descriptions unless necessary. Only necessary E.g. if the output value is int but can only return 3 specific values, document that.

## Commit Messages

After each implementation, write a concise commit message following this format:

**Title:** Short summary

**Body:**
- Brief explanation of what changed and why
- Keep it focused and to the point
- Dont include exact lines. Focus on what and why.

**Example:**
```
Split EmbeddingManager into Embedder and EnrollmentService

Separate embedding generation from enrollment data management for better separation of concerns.
EmbeddingManager violated SRP by handling both model inference and enrollment image processing.
```
