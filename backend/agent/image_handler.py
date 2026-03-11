"""
Image ingestion and multimodal routing for CreditScope.

Two processing paths:
1. OCR Path: For scanned documents (pay stubs, tax returns, bank statements)
2. Multimodal Path: For images requiring visual understanding
"""

from __future__ import annotations

import base64
import io
import re

import structlog

logger = structlog.get_logger(__name__)


class ImageHandler:
    """
    Handles image uploads containing customer/loan data.

    Routes images through OCR or multimodal processing based on content.
    """

    # Document type detection patterns
    DOCUMENT_KEYWORDS = [
        "pay stub", "paycheck", "earnings", "statement",
        "tax return", "1040", "w-2", "w2",
        "bank statement", "account summary",
        "deed", "property", "title",
    ]

    async def process_images(
        self, images: list[bytes], context: str = ""
    ) -> list[dict]:
        """
        Process uploaded images for the agent.

        Args:
            images: List of raw image bytes
            context: User query context to help route processing

        Returns:
            List of processed image data dicts ready for the model
        """
        processed = []
        for img_bytes in images:
            try:
                if self._looks_like_document(img_bytes, context):
                    result = self._ocr_extract(img_bytes, context)
                    processed.append(result)
                else:
                    b64 = base64.b64encode(img_bytes).decode()
                    processed.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
            except Exception as e:
                logger.error("image_processing_failed", error=str(e))
                b64 = base64.b64encode(img_bytes).decode()
                processed.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                })

        return processed

    def _looks_like_document(self, img_bytes: bytes, context: str) -> bool:
        """Determine if an image is likely a scannable document."""
        context_lower = context.lower()
        doc_keywords = [
            "pay stub", "tax return", "bank statement",
            "w-2", "1040", "deed", "document", "upload",
            "scan", "statement",
        ]
        return any(kw in context_lower for kw in doc_keywords)

    def _ocr_extract(self, img_bytes: bytes, context: str) -> dict:
        """Extract text from a document image via OCR."""
        try:
            from PIL import Image
            import pytesseract

            img = Image.open(io.BytesIO(img_bytes))
            text = pytesseract.image_to_string(img)

            structured = self._parse_document(text, context)
            return {
                "type": "extracted_data",
                "data": structured,
                "raw_text": text,
            }
        except ImportError:
            logger.warning("ocr_dependencies_unavailable")
            b64 = base64.b64encode(img_bytes).decode()
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }

    def _parse_document(self, text: str, context: str) -> dict:
        """Parse OCR text into structured fields based on document type."""
        doc_type = self._detect_document_type(text, context)
        data: dict = {"document_type": doc_type, "raw_text_preview": text[:500]}

        if doc_type == "pay_stub":
            data.update(self._parse_pay_stub(text))
        elif doc_type == "tax_return":
            data.update(self._parse_tax_return(text))
        elif doc_type == "bank_statement":
            data.update(self._parse_bank_statement(text))
        else:
            data["extracted_text"] = text

        return data

    def _detect_document_type(self, text: str, context: str) -> str:
        """Detect document type from OCR text and context."""
        combined = (text + " " + context).lower()

        if any(kw in combined for kw in ["pay stub", "paycheck", "earnings statement", "net pay"]):
            return "pay_stub"
        if any(kw in combined for kw in ["1040", "tax return", "adjusted gross", "filing status"]):
            return "tax_return"
        if any(kw in combined for kw in ["bank statement", "account summary", "beginning balance"]):
            return "bank_statement"
        if any(kw in combined for kw in ["deed", "property", "title", "parcel"]):
            return "property_deed"
        return "unknown"

    def _parse_pay_stub(self, text: str) -> dict:
        """Extract pay stub fields."""
        fields: dict = {}
        patterns = {
            "gross_pay": r"gross\s*(?:pay|earnings)[:\s]*\$?([\d,]+\.?\d*)",
            "net_pay": r"net\s*pay[:\s]*\$?([\d,]+\.?\d*)",
            "pay_period": r"pay\s*period[:\s]*(.+?)(?:\n|$)",
            "employer": r"(?:employer|company)[:\s]*(.+?)(?:\n|$)",
            "ytd_earnings": r"ytd\s*(?:gross|earnings)[:\s]*\$?([\d,]+\.?\d*)",
        }
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip().replace(",", "")
        return fields

    def _parse_tax_return(self, text: str) -> dict:
        """Extract tax return fields."""
        fields: dict = {}
        patterns = {
            "adjusted_gross_income": r"adjusted\s*gross\s*income[:\s]*\$?([\d,]+\.?\d*)",
            "total_income": r"total\s*income[:\s]*\$?([\d,]+\.?\d*)",
            "filing_status": r"filing\s*status[:\s]*(.+?)(?:\n|$)",
            "tax_year": r"(?:tax\s*year|form\s*1040)[:\s]*(\d{4})",
        }
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip().replace(",", "")
        return fields

    def _parse_bank_statement(self, text: str) -> dict:
        """Extract bank statement fields."""
        fields: dict = {}
        patterns = {
            "beginning_balance": r"beginning\s*balance[:\s]*\$?([\d,]+\.?\d*)",
            "ending_balance": r"ending\s*balance[:\s]*\$?([\d,]+\.?\d*)",
            "total_deposits": r"total\s*deposits[:\s]*\$?([\d,]+\.?\d*)",
            "total_withdrawals": r"total\s*withdrawals[:\s]*\$?([\d,]+\.?\d*)",
            "account_number": r"account[:\s#]*(\*{4,}\d{4}|\d{4,})",
        }
        for field_name, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                fields[field_name] = match.group(1).strip().replace(",", "")
        return fields
