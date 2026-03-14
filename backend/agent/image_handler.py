"""
Image ingestion for CreditScope.

Converts uploaded images to base64 data-URLs and forwards them directly
to the model for visual understanding.  No OCR / Tesseract dependency.
"""

from __future__ import annotations

import base64

import structlog

logger = structlog.get_logger(__name__)


class ImageHandler:
    """
    Converts raw image bytes into model-ready data-URL dicts.

    All visual analysis is delegated to the model's own vision capabilities.
    """

    async def process_images(
        self, images: list[bytes], context: str = ""
    ) -> list[dict]:
        """
        Encode every image as a base64 data-URL for the model.

        Args:
            images: List of raw image bytes.
            context: User query context (unused — kept for API compat).

        Returns:
            List of ``{"type": "image_url", "image_url": {"url": ...}}`` dicts.
        """
        processed: list[dict] = []
        for img_bytes in images:
            try:
                processed.append(self._as_image_url(img_bytes))
            except Exception as e:
                logger.error("image_encoding_failed", error=str(e))
                processed.append({
                    "type": "extracted_data",
                    "data": {
                        "error": "Failed to encode image",
                        "guidance": "The uploaded image could not be processed. "
                                    "Please try a different file.",
                    },
                })
        return processed

    # ── helpers ────────────────────────────────────────────────────────────

    def _as_image_url(self, img_bytes: bytes) -> dict:
        mime_type = self._detect_mime_type(img_bytes)
        b64 = base64.b64encode(img_bytes).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        }

    @staticmethod
    def _detect_mime_type(img_bytes: bytes) -> str:
        if img_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if img_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if img_bytes.startswith((b"II*\x00", b"MM\x00*")):
            return "image/tiff"
        if img_bytes.startswith(b"%PDF-"):
            return "application/pdf"
        return "image/jpeg"
