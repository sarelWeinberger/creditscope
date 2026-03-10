"""
Pydantic v2 schemas for Loan Application data.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LoanBase(BaseModel):
    loan_type: str
    requested_amount: float = Field(..., gt=0)
    term_months: int = Field(..., gt=0)
    collateral_type: str = "none"
    collateral_value: float = Field(0.0, ge=0)
    collateral_description: Optional[str] = None
    loan_purpose: Optional[str] = None

    model_config = {"from_attributes": True}


class LoanCreate(LoanBase):
    customer_id: int
    credit_score_at_application: Optional[int] = None
    dti_at_application: Optional[float] = None


class LoanResponse(LoanBase):
    id: int
    customer_id: int
    approved_amount: Optional[float] = None
    interest_rate: Optional[float] = None
    monthly_payment: Optional[float] = None
    status: str
    application_date: datetime
    decision_date: Optional[datetime] = None
    disbursement_date: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    underwriter_notes: Optional[str] = None
    credit_score_at_application: Optional[int] = None
    dti_at_application: Optional[float] = None
    risk_grade_at_application: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class LoanParams(BaseModel):
    """Parameters for loan-adjusted credit scoring."""
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in dollars")
    loan_type: str = Field(..., description="Type of loan (personal, mortgage, auto, etc.)")
    term_months: int = Field(..., gt=0, description="Loan term in months")
    collateral_type: str = Field("none", description="Type of collateral")
    collateral_value: float = Field(0.0, ge=0, description="Collateral value in dollars")
    additional_monthly_payment: Optional[float] = Field(
        None, ge=0, description="Monthly payment for new loan (computed if None)"
    )
    loan_purpose: Optional[str] = Field(None, description="Description of loan purpose")

    model_config = {"from_attributes": True}
