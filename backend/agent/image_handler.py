"""
Image and document processing handler.
Supports OCR via pytesseract and multimodal (base64) paths.
"""
import base64
import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageFilter, ImageStat
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


# Document type templates for structured parsing
DOCUMENT_TEMPLATES = {
    "pay_stub": {
        "fields": [
            "employee_name", "employer_name", "pay_period_start", "pay_period_end",
            "gross_pay", "net_pay", "federal_tax", "state_tax", "fica_tax",
            "year_to_date_gross", "pay_frequency",
        ],
        "patterns": {
            "gross_pay": r"(?:gross\s+pay|gross\s+earnings)[:\s]+\$?([\d,]+\.?\d*)",
            "net_pay": r"(?:net\s+pay|take.home)[:\s]+\$?([\d,]+\.?\d*)",
            "employer_name": r"^([A-Z][A-Za-z\s,\.]+(?:Inc|LLC|Corp|Ltd|Co)\.?)",
            "employee_name": r"(?:employee\s+name|pay\s+to)[:\s]+([A-Za-z\s]+)",
            "pay_period": r"(?:pay\s+period)[:\s]+(\d{1,2}/\d{1,2}/\d{4})\s*[-–]\s*(\d{1,2}/\d{1,2}/\d{4})",
        },
    },
    "tax_return": {
        "fields": [
            "taxpayer_name", "filing_year", "filing_status", "adjusted_gross_income",
            "taxable_income", "total_tax", "refund_amount", "wages_salaries",
            "business_income", "total_income",
        ],
        "patterns": {
            "adjusted_gross_income": r"(?:adjusted\s+gross\s+income|line\s+11|agi)[:\s]+\$?([\d,]+)",
            "taxable_income": r"(?:taxable\s+income|line\s+15)[:\s]+\$?([\d,]+)",
            "total_tax": r"(?:total\s+tax|line\s+24)[:\s]+\$?([\d,]+)",
            "filing_year": r"(?:tax\s+year|for\s+the\s+year)[:\s]+(\d{4})",
        },
    },
    "bank_statement": {
        "fields": [
            "account_holder", "account_number_last4", "statement_date",
            "beginning_balance", "ending_balance", "total_deposits", "total_withdrawals",
            "average_daily_balance",
        ],
        "patterns": {
            "beginning_balance": r"(?:beginning|opening|starting)\s+balance[:\s]+\$?([\d,]+\.?\d*)",
            "ending_balance": r"(?:ending|closing)\s+balance[:\s]+\$?([\d,]+\.?\d*)",
            "total_deposits": r"(?:total\s+deposits|total\s+credits)[:\s]+\$?([\d,]+\.?\d*)",
            "total_withdrawals": r"(?:total\s+withdrawals|total\s+debits)[:\s]+\$?([\d,]+\.?\d*)",
            "account_number": r"(?:account\s+(?:number|#|no\.?))[:\s]*[xX*]+(\d{4})",
        },
    },
    "government_id": {
        "fields": [
            "full_name", "date_of_birth", "id_number", "expiration_date",
            "address", "state", "id_type",
        ],
        "patterns": {
            "date_of_birth": r"(?:dob|date\s+of\s+birth|born)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "expiration": r"(?:exp(?:ires)?|expiration\s+date)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "id_number": r"(?:dl|license|id)\s*(?:no|number|#)?[:\s]+([A-Z0-9]+)",
        },
    },
    "property_deed": {
        "fields": [
            "property_address", "grantor", "grantee", "deed_date",
            "assessed_value", "legal_description", "parcel_number",
        ],
        "patterns": {
            "assessed_value": r"(?:assessed\s+value|appraised\s+value)[:\s]+\$?([\d,]+)",
            "parcel_number": r"(?:parcel\s+(?:id|number|no\.?|#))[:\s]+([0-9-]+)",
            "deed_date": r"(?:recorded|executed|dated)[:\s]+(\w+\s+\d{1,2},?\s+\d{4})",
        },
    },
}


class ImageHandler:
    """
    Handles image/document processing for the CreditScope agent.
    Supports OCR extraction and multimodal (base64) analysis.
    """

    MAX_IMAGES = 5
    MAX_IMAGE_SIZE = (2048, 2048)

    def __init__(self):
        self.has_pil = HAS_PIL
        self.has_tesseract = HAS_TESSERACT

    async def process_images(
        self, images: List[bytes], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of images and return structured data.

        Args:
            images: List of raw image bytes
            context: Context dict (customer_id, document_type hints, etc.)

        Returns:
            List of dicts with extracted data per image
        """
        results = []
        for i, img_bytes in enumerate(images[: self.MAX_IMAGES]):
            try:
                result = await self._process_single_image(img_bytes, i, context)
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "index": i,
                        "error": str(e),
                        "success": False,
                        "text": "",
                        "parsed_data": {},
                    }
                )
        return results

    async def _process_single_image(
        self, img_bytes: bytes, index: int, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single image file."""
        result: Dict[str, Any] = {
            "index": index,
            "success": True,
            "is_document": False,
            "document_type": None,
            "text": "",
            "parsed_data": {},
            "confidence": 0.0,
            "b64": None,
        }

        if not HAS_PIL:
            # Fallback: just encode as base64 for multimodal processing
            result["b64"] = base64.b64encode(img_bytes).decode("utf-8")
            result["method"] = "multimodal_only"
            return result

        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            raise ValueError(f"Cannot open image: {e}")

        # Resize if too large
        img.thumbnail(self.MAX_IMAGE_SIZE, Image.LANCZOS)

        # Detect if this looks like a document
        is_doc = self._looks_like_document(img)
        result["is_document"] = is_doc

        # Determine document type from context or content
        doc_type = context.get("document_type", "other")

        if is_doc and HAS_TESSERACT:
            # OCR path
            try:
                ocr_text = pytesseract.image_to_string(img)
                confidence_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                confidences = [c for c in confidence_data["conf"] if c > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0

                result["text"] = ocr_text
                result["confidence"] = round(avg_confidence / 100, 3)
                result["method"] = "ocr"

                # Infer document type from text if not provided
                if doc_type == "other":
                    doc_type = self._infer_document_type(ocr_text)

                result["document_type"] = doc_type
                result["parsed_data"] = self._parse_document(ocr_text, context, doc_type)

            except Exception as e:
                result["ocr_error"] = str(e)

        # Always include base64 for multimodal fallback
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result["b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        result["document_type"] = doc_type

        return result

    def _looks_like_document(self, img) -> bool:
        """
        Heuristic to determine if an image looks like a document.
        Documents tend to be:
          - Grayscale or low-color
          - High contrast (white background, dark text)
          - Portrait orientation
        """
        if not HAS_PIL:
            return False

        # Convert to grayscale for analysis
        gray = img.convert("L")
        stat = ImageStat.Stat(gray)

        mean_brightness = stat.mean[0]
        stddev = stat.stddev[0]

        # High mean (bright background) and moderate stddev (dark text on white)
        is_bright_bg = mean_brightness > 160
        has_text_contrast = 30 < stddev < 120

        # Check color diversity — documents have few colors
        rgb = img.convert("RGB")
        rgb_stat = ImageStat.Stat(rgb)
        color_variance = max(rgb_stat.stddev)
        is_low_color = color_variance < 80

        # Aspect ratio: documents are typically portrait or near-square
        w, h = img.size
        aspect = h / max(w, 1)
        is_document_aspect = 1.0 <= aspect <= 2.0

        score = sum([is_bright_bg, has_text_contrast, is_low_color, is_document_aspect])
        return score >= 3

    def _infer_document_type(self, text: str) -> str:
        """Guess document type from OCR text keywords."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["gross pay", "net pay", "pay period", "pay stub"]):
            return "pay_stub"
        if any(k in text_lower for k in ["form 1040", "adjusted gross income", "taxable income", "irs"]):
            return "tax_return"
        if any(k in text_lower for k in ["account balance", "statement date", "deposits", "withdrawals"]):
            return "bank_statement"
        if any(k in text_lower for k in ["license", "driver", "state id", "passport", "date of birth"]):
            return "government_id"
        if any(k in text_lower for k in ["deed", "grantor", "grantee", "parcel", "recorded"]):
            return "property_deed"
        return "other"

    def _parse_document(
        self, text: str, context: Dict[str, Any], doc_type: str
    ) -> Dict[str, Any]:
        """
        Extract structured fields from OCR text using document-type templates.
        """
        if doc_type not in DOCUMENT_TEMPLATES:
            return {"raw_text_length": len(text)}

        template = DOCUMENT_TEMPLATES[doc_type]
        extracted: Dict[str, Any] = {}

        for field_name, pattern in template.get("patterns", {}).items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                raw_value = match.group(1).strip().replace(",", "")
                # Try to convert to number if it looks like one
                try:
                    if "." in raw_value:
                        extracted[field_name] = float(raw_value)
                    else:
                        extracted[field_name] = int(raw_value)
                except ValueError:
                    extracted[field_name] = raw_value

        extracted["_document_type"] = doc_type
        extracted["_extraction_method"] = "regex_ocr"
        extracted["_text_length"] = len(text)

        return extracted

    def build_multimodal_message(
        self, images_data: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """
        Build a multimodal message for the model with text and image content.
        """
        content = [{"type": "text", "text": query}]

        for img_data in images_data:
            b64 = img_data.get("b64")
            if b64:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    }
                )

        return content
