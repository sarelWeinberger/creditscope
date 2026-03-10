"""
Customers router: CRUD and credit analysis endpoints.
"""
import json
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from db.queries import (
    get_customer_by_id,
    get_customer_documents,
    get_customer_loans,
    list_customers,
    search_customers,
)
from schemas.customer import CustomerListResponse, CustomerResponse
from schemas.loan import LoanResponse

router = APIRouter(prefix="/customers", tags=["customers"])


async def get_session():
    from main import SessionFactory
    async with SessionFactory() as session:
        yield session


@router.get("", response_model=CustomerListResponse)
async def list_customers_endpoint(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    min_score: Optional[int] = Query(None, ge=300, le=850),
    max_score: Optional[int] = Query(None, ge=300, le=850),
    employment_status: Optional[str] = None,
    search: Optional[str] = None,
    search_type: str = Query("fuzzy", regex="^(fuzzy|name|id|email)$"),
    session: AsyncSession = Depends(get_session),
):
    """
    List customers with pagination and optional filtering.
    Supports search by name, ID, or email.
    """
    if search:
        customers = await search_customers(session, search, search_type)
        total = len(customers)
        # Manual pagination for search results
        start = (page - 1) * page_size
        customers = customers[start : start + page_size]
    else:
        filters = {}
        if min_score is not None:
            filters["min_score"] = min_score
        if max_score is not None:
            filters["max_score"] = max_score
        if employment_status:
            filters["employment_status"] = employment_status

        customers, total = await list_customers(
            session, page=page, page_size=page_size, filters=filters
        )

    total_pages = max(1, (total + page_size - 1) // page_size)

    return CustomerListResponse(
        customers=[CustomerResponse.model_validate(c) for c in customers],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{customer_id}", response_model=CustomerResponse)
async def get_customer(
    customer_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get a single customer by ID."""
    customer = await get_customer_by_id(session, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")
    return CustomerResponse.model_validate(customer)


@router.get("/{customer_id}/credit-score")
async def get_credit_score(
    customer_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Calculate and return the credit score for a customer."""
    customer = await get_customer_by_id(session, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    from tools.credit_score import calculate_base_credit_score
    return calculate_base_credit_score(customer)


@router.get("/{customer_id}/loans", response_model=List[LoanResponse])
async def get_loans(
    customer_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get all loan applications for a customer."""
    customer = await get_customer_by_id(session, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    loans = await get_customer_loans(session, customer_id)
    return [LoanResponse.model_validate(loan) for loan in loans]


@router.get("/{customer_id}/documents")
async def get_documents(
    customer_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get all documents for a customer."""
    customer = await get_customer_by_id(session, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    docs = await get_customer_documents(session, customer_id)
    return [
        {
            "id": doc.id,
            "customer_id": doc.customer_id,
            "document_type": doc.document_type,
            "file_name": doc.file_name,
            "file_size_bytes": doc.file_size_bytes,
            "mime_type": doc.mime_type,
            "is_parsed": doc.is_parsed,
            "is_verified": doc.is_verified,
            "uploaded_at": doc.uploaded_at.isoformat() if doc.uploaded_at else None,
        }
        for doc in docs
    ]


@router.post("/{customer_id}/documents")
async def upload_document(
    customer_id: int,
    file: UploadFile = File(...),
    document_type: str = "other",
    session: AsyncSession = Depends(get_session),
):
    """
    Upload a document for a customer.
    Triggers OCR parsing if the file is an image or PDF.
    """
    customer = await get_customer_by_id(session, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    # Validate file type
    allowed_types = {"image/jpeg", "image/png", "application/pdf", "image/tiff"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}",
        )

    file_bytes = await file.read()
    file_size = len(file_bytes)

    # Save file to disk
    upload_dir = os.environ.get("UPLOAD_DIR", "/tmp/creditscope_uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{customer_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    # Parse document
    from agent.image_handler import ImageHandler
    handler = ImageHandler()
    parsed_results = await handler.process_images(
        [file_bytes], {"document_type": document_type}
    )

    parsed_data = None
    if parsed_results and parsed_results[0].get("parsed_data"):
        parsed_data = parsed_results[0]["parsed_data"]

    # Create document record
    from db.queries import create_customer_document
    doc_data = {
        "customer_id": customer_id,
        "document_type": document_type,
        "file_name": file.filename or "document",
        "file_path": file_path,
        "file_size_bytes": file_size,
        "mime_type": file.content_type,
        "is_parsed": parsed_data is not None,
        "parsed_data": json.dumps(parsed_data) if parsed_data else None,
    }

    doc = await create_customer_document(session, doc_data)
    await session.commit()

    return {
        "document_id": doc.id,
        "customer_id": customer_id,
        "document_type": document_type,
        "file_name": file.filename,
        "file_size_bytes": file_size,
        "is_parsed": doc.is_parsed,
        "parsed_fields": list(parsed_data.keys()) if parsed_data else [],
        "status": "uploaded",
    }


@router.get("/{customer_id}/risk-assessment")
async def get_risk_assessment(
    customer_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get comprehensive risk assessment for a customer."""
    customer = await get_customer_by_id(session, customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_id} not found")

    from tools.risk_adjustment import compute_risk_weighted_score
    return compute_risk_weighted_score(customer)
