# Claude Code Development Instructions

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
