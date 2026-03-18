"""Image generation tool for agents.

Routes through the mesh API proxy so agents never hold API keys.
Supports Gemini (default) and OpenAI DALL-E 3 providers.
Generated images are saved as artifacts and returned as multimodal
content blocks so the LLM can see and iterate on them.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path

from src.agent.skills import skill
from src.shared.utils import setup_logging

logger = setup_logging("agent.image_gen")

_VALID_SIZES = {"square", "landscape", "portrait"}
_VALID_PROVIDERS = {"gemini", "openai"}
_ARTIFACTS_DIR = Path("/data/workspace/artifacts")
_FILENAME_UNSAFE_RE = re.compile(r"[^a-zA-Z0-9._-]")


@skill(
    name="generate_image",
    description=(
        "Generate an image from a text prompt using AI image generation. "
        "The image is saved to the workspace artifacts directory and returned "
        "visually so you can see the result. Supports 'gemini' (default) and "
        "'openai' providers."
    ),
    parameters={
        "prompt": {
            "type": "string",
            "description": "Text description of the image to generate",
        },
        "size": {
            "type": "string",
            "description": "Image aspect ratio: 'square' (default), 'landscape', or 'portrait'",
            "enum": ["square", "landscape", "portrait"],
            "default": "square",
        },
        "filename": {
            "type": "string",
            "description": "Output filename (without path). Auto-generated if not provided.",
        },
        "provider": {
            "type": "string",
            "description": "Image generation provider: 'gemini' (default) or 'openai'",
            "enum": ["gemini", "openai"],
            "default": "gemini",
        },
    },
)
async def generate_image(
    prompt: str,
    size: str = "square",
    filename: str = "",
    provider: str = "gemini",
    *,
    mesh_client=None,
    workspace_manager=None,
) -> dict:
    """Generate an image and save it as an artifact."""
    if not mesh_client:
        return {"error": "Image generation requires mesh connectivity"}

    if not prompt or not prompt.strip():
        return {"error": "prompt is required"}

    if size not in _VALID_SIZES:
        size = "square"

    if provider not in _VALID_PROVIDERS:
        provider = "gemini"

    # Generate filename if not provided
    if not filename:
        slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower().strip())[:40].strip("_")
        filename = f"{slug}.png" if slug else "generated_image.png"
    else:
        # Sanitize user-provided filename: strip path components, keep safe chars
        filename = Path(filename).name  # strip any directory components
        filename = _FILENAME_UNSAFE_RE.sub("_", filename)

    # Ensure filename has an extension
    if "." not in filename:
        filename += ".png"

    # Final safety check: ensure resolved path stays within artifacts dir
    resolved = (_ARTIFACTS_DIR / filename).resolve()
    if not str(resolved).startswith(str(_ARTIFACTS_DIR.resolve())):
        return {"error": "Invalid filename — path traversal not allowed"}

    try:
        result = await mesh_client.image_generate(
            prompt=prompt, size=size, provider=provider, timeout=60,
        )
    except Exception as e:
        logger.error("Image generation failed: %s", e)
        return {"error": f"Image generation request failed: {e}"}

    if not result.get("success"):
        error = result.get("error", "Unknown error")
        return {"error": f"Image generation failed: {error}"}

    data = result.get("data", {})
    image_base64 = data.get("image_base64", "")
    mime_type = data.get("mime_type", "image/png")

    if not image_base64:
        return {"error": "No image data returned from provider"}

    # Save to artifacts directory
    try:
        _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        filepath = _ARTIFACTS_DIR / filename
        filepath.write_bytes(base64.b64decode(image_base64))
        logger.info("Image saved to %s", filepath)
    except Exception as e:
        logger.error("Failed to save image: %s", e)
        return {"error": f"Image generated but failed to save: {e}"}

    # Register as artifact on blackboard if possible
    if workspace_manager and mesh_client.project_name:
        try:
            await mesh_client.write_blackboard(
                f"artifacts/{filename}",
                {
                    "type": "image",
                    "path": str(filepath),
                    "prompt": prompt[:200],
                    "provider": provider,
                    "agent": mesh_client.agent_id,
                },
            )
        except Exception:
            pass  # Non-critical — image is already saved

    return {
        "status": "image generated",
        "path": str(filepath),
        "provider": provider,
        "_image": {"data": image_base64, "media_type": mime_type},
    }
