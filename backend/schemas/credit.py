"""
Pydantic v2 schemas for credit scoring responses.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CreditScoreResponse(BaseModel):
    """FICO-like credit score breakdown."""
    customer_id: int
    score: int = Field(..., ge=300, le=850)
    grade: str  # A-F
    grade_label: str  # Excellent, Good, Fair, Poor, Very Poor, Extremely Poor

    # Factor scores (0-100 each)
    payment_history_score: float
    credit_utilization_score: float
    credit_age_score: float
    credit_mix_score: float
    new_credit_score: float

    # Factor weights (sum to 100)
    payment_history_weight: float = 35.0
    credit_utilization_weight: float = 30.0
    credit_age_weight: float = 15.0
    credit_mix_weight: float = 10.0
    new_credit_weight: float = 10.0

    # Explanations
    factors: Dict[str, str]  # factor_name -> explanation text
    breakdown: Dict[str, Any]  # detailed numeric breakdown
    recommendations: List[str]  # actionable improvement tips

    model_config = {"from_attributes": True}


class LoanAdjustedScoreResponse(BaseModel):
    """Credit score adjusted for a specific loan application."""
    customer_id: int
    base_score: int
    adjusted_score: int
    final_grade: str
    final_grade_label: str

    # Individual adjustments
    dti_adjustment: float
    collateral_adjustment: float
    term_risk_adjustment: float
    purpose_adjustment: float
    total_adjustment: float

    # Explanations
    adjustments: Dict[str, str]  # adjustment_name -> explanation
    loan_recommendation: str  # approve / conditional / deny

    model_config = {"from_attributes": True}


class DTIResponse(BaseModel):
    """Debt-to-income ratio analysis."""
    customer_id: int
    monthly_income: float
    monthly_debt_payments: float
    additional_payment: float

    front_end_ratio: float  # housing costs / income
    back_end_ratio: float   # all debt / income
    combined_dti: float     # total debt after new loan / income

    risk_classification: str  # excellent/good/fair/poor/very_poor
    risk_label: str
    impact_description: str
    thresholds: Dict[str, float]

    model_config = {"from_attributes": True}


class CollateralResponse(BaseModel):
    """Collateral evaluation result."""
    collateral_type: str
    collateral_value: float
    loan_amount: float

    ltv_ratio: float         # loan / collateral value
    coverage_ratio: float    # collateral / loan
    haircut_pct: float       # depreciation / risk haircut
    adjusted_value: float    # collateral_value * (1 - haircut)

    risk_assessment: str     # low/medium/high/very_high
    risk_description: str
    score_impact: float      # points added to score (can be negative)

    model_config = {"from_attributes": True}


class PaymentHistoryResponse(BaseModel):
    """Payment history analysis."""
    customer_id: int
    on_time_payments: int
    total_late_events: int

    late_30d_count: int
    late_60d_count: int
    late_90d_count: int
    collections_count: int
    bankruptcies_count: int
    foreclosures_count: int
    charge_offs_count: int

    on_time_rate: float      # 0.0 - 1.0
    delinquency_score: float # 0-100 (higher = worse)
    severity_score: float    # weighted severity
    trend: str               # improving / stable / deteriorating
    trend_description: str
    score_impact: float      # net impact on credit score

    model_config = {"from_attributes": True}


class RiskWeightedResponse(BaseModel):
    """Comprehensive risk-weighted credit assessment."""
    customer_id: int

    base_score: int
    payment_history_contribution: float
    utilization_contribution: float
    credit_age_contribution: float
    credit_mix_contribution: float
    new_credit_contribution: float

    risk_score: float        # composite 0-1000
    risk_grade: str          # A through F
    risk_label: str
    recommendation: str      # Approve / Conditional Approval / Review / Decline

    breakdown: Dict[str, Any]
    key_risk_factors: List[str]
    positive_factors: List[str]

    model_config = {"from_attributes": True}


class ExecutionStep(BaseModel):
    """Single step in the agent's ReAct loop."""
    step: int
    tool_name: str
    tool_input: Dict[str, Any]
    tool_output: Any
    duration_ms: float
    success: bool
    error: Optional[str] = None


class AgentResponse(BaseModel):
    """Full response from the CreditScope agent."""
    session_id: str
    answer: str
    execution_trace: List[ExecutionStep] = []
    moe_traces: Optional[Dict[str, Any]] = None
    tokens_used: Dict[str, int] = Field(default_factory=dict)
    thinking: Optional[Dict[str, Any]] = None
    total_duration_ms: float = 0.0
    model: str = "creditscope"

    model_config = {"from_attributes": True}
