"""Customer lookup and search queries for CreditScope."""

from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.db.models import Customer, CustomerDocument, LoanApplication


def lookup_customer(db: Session, query: str, search_type: str = "fuzzy") -> list[Customer]:
    """Look up customers by name, ID, or fuzzy match."""
    if search_type == "id":
        try:
            cid = int(query)
            customer = db.query(Customer).filter(Customer.id == cid).first()
            return [customer] if customer else []
        except ValueError:
            return []

    if search_type == "name":
        return (
            db.query(Customer)
            .filter(Customer.full_name.ilike(f"%{query}%"))
            .all()
        )

    # Fuzzy: search by name, SSN last 4, or ID
    try:
        cid = int(query)
        by_id = db.query(Customer).filter(Customer.id == cid).first()
        if by_id:
            return [by_id]
    except ValueError:
        pass

    return (
        db.query(Customer)
        .filter(
            or_(
                Customer.full_name.ilike(f"%{query}%"),
                Customer.ssn_last4 == query[-4:] if len(query) >= 4 else False,
            )
        )
        .limit(10)
        .all()
    )


def get_customer_by_id(db: Session, customer_id: int) -> Customer | None:
    """Get a customer by primary key."""
    return db.query(Customer).filter(Customer.id == customer_id).first()


def list_customers(db: Session, skip: int = 0, limit: int = 20) -> list[Customer]:
    """List customers with pagination."""
    return db.query(Customer).offset(skip).limit(limit).all()


def count_customers(db: Session) -> int:
    """Count total customers."""
    return db.query(Customer).count()


def get_customer_loans(db: Session, customer_id: int) -> list[LoanApplication]:
    """Get all loan applications for a customer."""
    return (
        db.query(LoanApplication)
        .filter(LoanApplication.customer_id == customer_id)
        .all()
    )


def get_loan_by_id(db: Session, loan_id: int) -> LoanApplication | None:
    """Get a loan application by ID."""
    return db.query(LoanApplication).filter(LoanApplication.id == loan_id).first()


def get_customer_documents(db: Session, customer_id: int) -> list[CustomerDocument]:
    """Get all documents for a customer."""
    return (
        db.query(CustomerDocument)
        .filter(CustomerDocument.customer_id == customer_id)
        .all()
    )


def create_document(
    db: Session,
    customer_id: int,
    document_type: str,
    file_path: str,
    extracted_data: dict | None = None,
) -> CustomerDocument:
    """Create a new customer document record."""
    doc = CustomerDocument(
        customer_id=customer_id,
        document_type=document_type,
        file_path=file_path,
        extracted_data=extracted_data,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc
