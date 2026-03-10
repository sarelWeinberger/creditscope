"""
Pydantic v2 schemas for Customer data.
"""
from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field, model_validator


class CustomerBase(BaseModel):
    first_name: str = Field(..., max_length=100)
    last_name: str = Field(..., max_length=100)
    email: str = Field(..., max_length=255)
    phone: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[date] = None
    ssn_last4: Optional[str] = Field(None, max_length=4)
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = Field(None, max_length=2)
    zip_code: Optional[str] = None

    # Employment
    employment_status: str = "employed"
    employer_name: Optional[str] = None
    annual_income: float = Field(0.0, ge=0)
    monthly_income: float = Field(0.0, ge=0)
    years_employed: Optional[float] = Field(None, ge=0)
    job_title: Optional[str] = None

    # Credit Profile
    credit_score: int = Field(650, ge=300, le=850)
    credit_history_years: float = Field(0.0, ge=0)
    num_credit_accounts: int = Field(0, ge=0)
    num_open_accounts: int = Field(0, ge=0)
    num_closed_accounts: int = Field(0, ge=0)
    total_credit_limit: float = Field(0.0, ge=0)
    total_credit_used: float = Field(0.0, ge=0)

    # Debt
    total_debt: float = Field(0.0, ge=0)
    monthly_debt_payments: float = Field(0.0, ge=0)
    mortgage_balance: float = Field(0.0, ge=0)
    auto_loan_balance: float = Field(0.0, ge=0)
    student_loan_balance: float = Field(0.0, ge=0)
    credit_card_balance: float = Field(0.0, ge=0)
    other_debt: float = Field(0.0, ge=0)

    # Payment History
    on_time_payments: int = Field(0, ge=0)
    late_payments_30d: int = Field(0, ge=0)
    late_payments_60d: int = Field(0, ge=0)
    late_payments_90d: int = Field(0, ge=0)
    collections: int = Field(0, ge=0)
    bankruptcies: int = Field(0, ge=0)
    foreclosures: int = Field(0, ge=0)
    charge_offs: int = Field(0, ge=0)

    # Inquiries
    hard_inquiries_6m: int = Field(0, ge=0)
    hard_inquiries_12m: int = Field(0, ge=0)
    soft_inquiries_12m: int = Field(0, ge=0)

    # Assets
    checking_balance: float = Field(0.0, ge=0)
    savings_balance: float = Field(0.0, ge=0)
    investment_balance: float = Field(0.0, ge=0)
    property_value: float = Field(0.0, ge=0)
    vehicle_value: float = Field(0.0, ge=0)

    notes: Optional[str] = None

    model_config = {"from_attributes": True}


class CustomerCreate(CustomerBase):
    """Schema for creating a new customer."""
    pass


class CustomerResponse(CustomerBase):
    """Schema for returning customer data."""
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @computed_field
    @property
    def credit_utilization_ratio(self) -> float:
        if self.total_credit_limit == 0:
            return 0.0
        return round(self.total_credit_used / self.total_credit_limit, 4)

    @computed_field
    @property
    def computed_total_debt(self) -> float:
        return round(
            self.mortgage_balance
            + self.auto_loan_balance
            + self.student_loan_balance
            + self.credit_card_balance
            + self.other_debt,
            2,
        )

    @computed_field
    @property
    def dti_ratio(self) -> float:
        if self.monthly_income == 0:
            return 0.0
        return round(self.monthly_debt_payments / self.monthly_income, 4)

    @computed_field
    @property
    def credit_grade(self) -> str:
        score = self.credit_score
        if score >= 750:
            return "A"
        elif score >= 700:
            return "B"
        elif score >= 650:
            return "C"
        elif score >= 600:
            return "D"
        elif score >= 550:
            return "E"
        return "F"

    model_config = {"from_attributes": True}


class CustomerListResponse(BaseModel):
    customers: List[CustomerResponse]
    total: int
    page: int
    page_size: int
    total_pages: int

    model_config = {"from_attributes": True}
