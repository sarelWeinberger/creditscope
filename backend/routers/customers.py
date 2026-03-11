"""
Customer CRUD endpoints.
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.db.models import get_session
from backend.db.queries import (
    get_customer_by_id,
    list_customers,
    get_customer_loans,
    get_customer_documents,
    add_customer_document,
)
from backend.schemas.customer import (
    CustomerResponse,
    CustomerListResponse,
    DocumentUploadResponse,
)
from backend.schemas.loan import LoanApplicationResponse
from backend.tools.credit_score import calculate_base_credit_score

router = APIRouter()


@router.get("/customers", response_model=CustomerListResponse)
async def list_all_customers(offset: int = 0, limit: int = 20, sort_by: str = "id"):
    """List all customers with pagination."""
    session = get_session()
    try:
        custs, total = list_customers(session, offset=offset, limit=limit, sort_by=sort_by)
        return CustomerListResponse(
            customers=[CustomerResponse.model_validate(c) for c in custs],
            total=total,
            offset=offset,
            limit=limit,
        )
    finally:
        session.close()


@router.get("/customers/{customer_id}", response_model=CustomerResponse)
async def get_customer(customer_id: int):
    """Get customer details by ID."""
    session = get_session()
    try:
        customer = get_customer_by_id(session, customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        return CustomerResponse.model_validate(customer)
    finally:
        session.close()


@router.get("/customers/{customer_id}/credit-score")
async def get_customer_credit_score(customer_id: int):
    """Get pre-computed credit score for a customer."""
    session = get_session()
    try:
        customer = get_customer_by_id(session, customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        return calculate_base_credit_score(customer)
    finally:
        session.close()


@router.get("/customers/{customer_id}/loans", response_model=list[LoanApplicationResponse])
async def get_loans(customer_id: int):
    """Get all loan applications for a customer."""
    session = get_session()
    try:
        customer = get_customer_by_id(session, customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")
        loans = get_customer_loans(session, customer_id)
        return [LoanApplicationResponse.model_validate(l) for l in loans]
    finally:
        session.close()


@router.post("/customers/{customer_id}/documents", response_model=DocumentUploadResponse)
async def upload_document(
    customer_id: int,
    document_type: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload a document image for a customer."""
    session = get_session()
    try:
        customer = get_customer_by_id(session, customer_id)
        if not customer:
            raise HTTPException(status_code=404, detail="Customer not found")

        contents = await file.read()

        # OCR extraction attempt
        extracted_data = None
        try:
            from backend.agent.image_handler import ImageHandler
            handler = ImageHandler()
            results = await handler.process_images([contents], context=document_type)
            if results and results[0].get("type") == "extracted_data":
                extracted_data = results[0]["data"]
        except Exception:
            pass

        doc = add_customer_document(
            session, customer_id, document_type,
            file_path=f"uploads/{file.filename}",
            extracted_data=extracted_data,
        )
        return DocumentUploadResponse.model_validate(doc)
    finally:
        session.close()
