"""Pydantic models for credit score responses."""

from pydantic import BaseModel, Field


class CreditScoreBreakdown(BaseModel):
    payment_history: float = Field(description="Payment history component (35%)")
    amounts_owed: float = Field(description="Amounts owed component (30%)")
    length_of_history: float = Field(description="Credit history length component (15%)")
    new_credit: float = Field(description="New credit component (10%)")
    credit_mix: float = Field(description="Credit mix component (10%)")


class CreditScoreResponse(BaseModel):
    score: int = Field(ge=300, le=850)
    grade: str
    factors: list[str]
    breakdown: CreditScoreBreakdown


class LoanAdjustedScoreResponse(BaseModel):
    base_score: int
    adjusted_score: int
    grade: str
    adjustments: dict[str, float]
    dti_ratio: float
    ltv_ratio: float | None = None
    risk_factors: list[str]
    recommendation: str


class DTIResponse(BaseModel):
    current_dti: float
    projected_dti: float | None = None
    gross_monthly_income: float
    total_monthly_debt: float
    additional_payment: float = 0
    assessment: str


class CollateralEvaluation(BaseModel):
    collateral_type: str
    collateral_value: float
    loan_amount: float
    ltv_ratio: float
    coverage_ratio: float
    risk_assessment: str
    score_adjustment: int


class PaymentHistoryAnalysis(BaseModel):
    delinquency_trend: str
    severity_score: float
    recovery_pattern: str
    late_payment_details: dict
    risk_level: str


class RiskWeightedScore(BaseModel):
    composite_score: int
    risk_grade: str
    component_scores: dict[str, float]
    risk_factors: list[str]
    recommendation: str
    confidence: float
