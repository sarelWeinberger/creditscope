"""
SQLAlchemy ORM models for CreditScope customer database.
"""

import os
from datetime import date, datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, Date, DateTime, Float, ForeignKey,
    Integer, JSON, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, Session


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(200), nullable=False, index=True)
    ssn_last4 = Column(String(4), nullable=False)
    date_of_birth = Column(Date, nullable=False)
    employment_status = Column(String(50), nullable=False)  # employed, self_employed, unemployed, retired
    employer_name = Column(String(200), nullable=True)
    annual_income = Column(Float, nullable=False)
    monthly_expenses = Column(Float, nullable=False)
    years_at_current_job = Column(Integer, nullable=False, default=0)
    residential_status = Column(String(50), nullable=False)  # own, rent, mortgage
    years_at_address = Column(Integer, nullable=False, default=0)
    credit_history_years = Column(Integer, nullable=False, default=0)
    num_open_accounts = Column(Integer, nullable=False, default=0)
    num_credit_cards = Column(Integer, nullable=False, default=0)
    total_credit_limit = Column(Float, nullable=False, default=0)
    total_credit_used = Column(Float, nullable=False, default=0)
    num_late_payments_12m = Column(Integer, nullable=False, default=0)
    num_late_payments_24m = Column(Integer, nullable=False, default=0)
    num_defaults = Column(Integer, nullable=False, default=0)
    num_bankruptcies = Column(Integer, nullable=False, default=0)
    num_collections = Column(Integer, nullable=False, default=0)
    has_mortgage = Column(Boolean, nullable=False, default=False)
    mortgage_balance = Column(Float, nullable=False, default=0)
    has_auto_loan = Column(Boolean, nullable=False, default=False)
    auto_loan_balance = Column(Float, nullable=False, default=0)
    has_student_loan = Column(Boolean, nullable=False, default=False)
    student_loan_balance = Column(Float, nullable=False, default=0)
    total_revolving_debt = Column(Float, nullable=False, default=0)
    monthly_debt_payments = Column(Float, nullable=False, default=0)
    num_hard_inquiries_6m = Column(Integer, nullable=False, default=0)
    num_hard_inquiries_12m = Column(Integer, nullable=False, default=0)
    fico_score = Column(Integer, nullable=False)  # Pre-computed base FICO 300-850
    risk_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    loan_applications = relationship("LoanApplication", back_populates="customer", cascade="all, delete-orphan")
    documents = relationship("CustomerDocument", back_populates="customer", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Customer(id={self.id}, name='{self.full_name}', fico={self.fico_score})>"


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    loan_type = Column(String(50), nullable=False)  # personal, mortgage, auto, business, student
    requested_amount = Column(Float, nullable=False)
    proposed_term_months = Column(Integer, nullable=False)
    proposed_interest_rate = Column(Float, nullable=True)
    collateral_type = Column(String(50), nullable=True)  # real_estate, vehicle, equipment, none
    collateral_value = Column(Float, nullable=True)
    purpose = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, default="pending")  # pending, approved, rejected, under_review
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="loan_applications")

    def __repr__(self):
        return f"<LoanApplication(id={self.id}, type='{self.loan_type}', amount={self.requested_amount})>"


class CustomerDocument(Base):
    __tablename__ = "customer_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, ForeignKey("customers.id"), nullable=False, index=True)
    document_type = Column(String(50), nullable=False)  # pay_stub, tax_return, bank_statement, id, property_deed
    file_path = Column(String(500), nullable=False)
    extracted_data = Column(JSON, nullable=True)
    uploaded_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    customer = relationship("Customer", back_populates="documents")

    def __repr__(self):
        return f"<CustomerDocument(id={self.id}, type='{self.document_type}')>"


# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/creditscope.db")

engine = create_engine(DATABASE_URL, echo=False)

from sqlalchemy.orm import sessionmaker as _sessionmaker

_SessionLocal = _sessionmaker(bind=engine)


def init_db():
    """Create all tables."""
    if DATABASE_URL.startswith("sqlite:///"):
        db_path = DATABASE_URL.replace("sqlite:///", "", 1)
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Get a new database session."""
    return _SessionLocal()
