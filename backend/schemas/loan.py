"""Pydantic models for loan structures."""

from datetime import datetime
from pydantic import BaseModel, Field


class LoanApplicationBase(BaseModel):
    loan_type: str
    requested_amount: float = Field(gt=0)
    proposed_term_months: int = Field(gt=0)
    proposed_interest_rate: float | None = None
    collateral_type: str | None = None
    collateral_value: float | None = None
    purpose: str


class LoanApplicationCreate(LoanApplicationBase):
    customer_id: int


class LoanApplicationResponse(LoanApplicationBase):
    id: int
    customer_id: int
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class LoanAdjustedScoreRequest(BaseModel):
    customer_id: int
    loan_amount: float = Field(gt=0)
    loan_term_months: int = Field(gt=0)
    interest_rate: float = Field(gt=0)
    loan_type: str
    collateral_type: str = "none"
    collateral_value: float = 0
