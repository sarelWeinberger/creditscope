"""
Customer lookup and search queries for CreditScope.
"""

from sqlalchemy import or_, func
from sqlalchemy.orm import Session

from backend.db.models import Customer, LoanApplication, CustomerDocument


def get_customer_by_id(session: Session, customer_id: int) -> Customer | None:
    """Get a customer by their ID."""
    return session.query(Customer).filter(Customer.id == customer_id).first()


def search_customers(
    session: Session,
    query: str,
    search_type: str = "fuzzy",
    limit: int = 10,
) -> list[Customer]:
    """
    Search for customers by name, ID, or fuzzy match.

    Args:
        session: Database session
        query: Search term
        search_type: "name", "id", or "fuzzy"
        limit: Maximum results to return
    """
    if search_type == "id":
        try:
            cid = int(query)
            customer = get_customer_by_id(session, cid)
            return [customer] if customer else []
        except ValueError:
            return []

    if search_type == "name":
        return (
            session.query(Customer)
            .filter(Customer.full_name.ilike(f"%{query}%"))
            .limit(limit)
            .all()
        )

    # Fuzzy: search by name or partial SSN
    return (
        session.query(Customer)
        .filter(
            or_(
                Customer.full_name.ilike(f"%{query}%"),
                Customer.ssn_last4.like(f"%{query}%"),
                Customer.id == _safe_int(query),
            )
        )
        .limit(limit)
        .all()
    )


def get_customer_loans(session: Session, customer_id: int) -> list[LoanApplication]:
    """Get all loan applications for a customer."""
    return (
        session.query(LoanApplication)
        .filter(LoanApplication.customer_id == customer_id)
        .order_by(LoanApplication.created_at.desc())
        .all()
    )


def get_loan_by_id(session: Session, loan_id: int) -> LoanApplication | None:
    """Get a loan application by ID."""
    return session.query(LoanApplication).filter(LoanApplication.id == loan_id).first()


def get_customer_documents(session: Session, customer_id: int) -> list[CustomerDocument]:
    """Get all documents for a customer."""
    return (
        session.query(CustomerDocument)
        .filter(CustomerDocument.customer_id == customer_id)
        .order_by(CustomerDocument.uploaded_at.desc())
        .all()
    )


def list_customers(
    session: Session,
    offset: int = 0,
    limit: int = 20,
    sort_by: str = "id",
) -> tuple[list[Customer], int]:
    """List customers with pagination. Returns (customers, total_count)."""
    total = session.query(func.count(Customer.id)).scalar() or 0

    sort_col = getattr(Customer, sort_by, Customer.id)
    customers = (
        session.query(Customer)
        .order_by(sort_col)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return customers, total


def add_customer_document(
    session: Session,
    customer_id: int,
    document_type: str,
    file_path: str,
    extracted_data: dict | None = None,
) -> CustomerDocument:
    """Add a document record for a customer."""
    doc = CustomerDocument(
        customer_id=customer_id,
        document_type=document_type,
        file_path=file_path,
        extracted_data=extracted_data,
    )
    session.add(doc)
    session.commit()
    return doc


def _safe_int(val: str) -> int:
    """Safely convert a string to int, returning -1 on failure."""
    try:
        return int(val)
    except (ValueError, TypeError):
        return -1
