"""Pydantic models for customer data."""

from datetime import date, datetime
from pydantic import BaseModel, Field


class CustomerBase(BaseModel):
    full_name: str
    ssn_last4: str = Field(min_length=4, max_length=4)
    date_of_birth: date
    employment_status: str
    employer_name: str | None = None
    annual_income: float
    monthly_expenses: float
    years_at_current_job: int = 0
    residential_status: str
    years_at_address: int = 0


class CustomerCreditProfile(BaseModel):
    credit_history_years: int
    num_open_accounts: int
    num_credit_cards: int
    total_credit_limit: float
    total_credit_used: float
    num_late_payments_12m: int
    num_late_payments_24m: int
    num_defaults: int
    num_bankruptcies: int
    num_collections: int
    has_mortgage: bool
    mortgage_balance: float
    has_auto_loan: bool
    auto_loan_balance: float
    has_student_loan: bool
    student_loan_balance: float
    total_revolving_debt: float
    monthly_debt_payments: float
    num_hard_inquiries_6m: int
    num_hard_inquiries_12m: int
    fico_score: int = Field(ge=300, le=850)
    risk_notes: str | None = None


class CustomerResponse(CustomerBase, CustomerCreditProfile):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CustomerListResponse(BaseModel):
    customers: list[CustomerResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CustomerSearchRequest(BaseModel):
    query: str
    search_type: str = "fuzzy"


class DocumentUploadResponse(BaseModel):
    id: int
    customer_id: int
    document_type: str
    extracted_data: dict | None = None
    uploaded_at: datetime

    model_config = {"from_attributes": True}
