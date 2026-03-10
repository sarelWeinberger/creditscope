"""
Debt-to-Income ratio calculator.
Computes front-end and back-end DTI ratios with risk classification.
"""
from typing import Any, Dict


DTI_THRESHOLDS = {
    "excellent": 0.28,
    "good": 0.36,
    "fair": 0.43,
    "poor": 0.50,
    "very_poor": 1.00,
}

RISK_LABELS = {
    "excellent": "Excellent — Well below standard lending thresholds.",
    "good": "Good — Within conventional lending guidelines.",
    "fair": "Fair — At or near FHA/standard lending limits.",
    "poor": "Poor — Above standard thresholds; higher-risk borrower.",
    "very_poor": "Very Poor — DTI significantly exceeds safe lending limits.",
}


def _classify_dti(ratio: float) -> tuple[str, str]:
    """Return (risk_classification, risk_label) for a DTI ratio."""
    if ratio <= DTI_THRESHOLDS["excellent"]:
        return "excellent", RISK_LABELS["excellent"]
    elif ratio <= DTI_THRESHOLDS["good"]:
        return "good", RISK_LABELS["good"]
    elif ratio <= DTI_THRESHOLDS["fair"]:
        return "fair", RISK_LABELS["fair"]
    elif ratio <= DTI_THRESHOLDS["poor"]:
        return "poor", RISK_LABELS["poor"]
    else:
        return "very_poor", RISK_LABELS["very_poor"]


def calculate_dti_ratio(customer, additional_monthly_payment: float = 0.0) -> Dict[str, Any]:
    """
    Calculate DTI ratios for a customer.

    Front-end ratio: housing costs (mortgage) / gross monthly income
    Back-end ratio: all monthly debt payments / gross monthly income
    Combined DTI: all debt including new loan payment / income

    Args:
        customer: Customer ORM model instance
        additional_monthly_payment: Monthly payment for a proposed new loan

    Returns:
        dict matching DTIResponse schema
    """
    monthly_income = customer.monthly_income or 0.0

    if monthly_income <= 0:
        return {
            "customer_id": customer.id,
            "monthly_income": 0.0,
            "monthly_debt_payments": customer.monthly_debt_payments,
            "additional_payment": additional_monthly_payment,
            "front_end_ratio": 0.0,
            "back_end_ratio": 0.0,
            "combined_dti": 0.0,
            "risk_classification": "very_poor",
            "risk_label": "Cannot calculate DTI: no monthly income on record.",
            "impact_description": "Income information is required for DTI calculation.",
            "thresholds": DTI_THRESHOLDS,
        }

    # Front-end: housing costs only
    # Approximate monthly mortgage payment from balance at 6% / 30yr
    monthly_mortgage = 0.0
    if customer.mortgage_balance > 0:
        rate = 0.06 / 12
        term = 360
        monthly_mortgage = customer.mortgage_balance * rate / (1 - (1 + rate) ** (-term))

    front_end = monthly_mortgage / monthly_income

    # Back-end: all current monthly debt payments
    back_end = customer.monthly_debt_payments / monthly_income

    # Combined: back-end + proposed new payment
    combined = (customer.monthly_debt_payments + additional_monthly_payment) / monthly_income

    risk_class, risk_label = _classify_dti(combined)

    # Impact description
    if combined <= 0.28:
        impact = "Low DTI — strong ability to service additional debt."
    elif combined <= 0.36:
        impact = "Acceptable DTI — generally qualifies for conventional loans."
    elif combined <= 0.43:
        impact = "Borderline DTI — may qualify for FHA loans; conventional lenders may add conditions."
    elif combined <= 0.50:
        impact = "High DTI — many lenders will decline or require compensating factors."
    else:
        impact = "Excessive DTI — very limited lending options; debt reduction recommended."

    return {
        "customer_id": customer.id,
        "monthly_income": round(monthly_income, 2),
        "monthly_debt_payments": round(customer.monthly_debt_payments, 2),
        "additional_payment": round(additional_monthly_payment, 2),
        "front_end_ratio": round(front_end, 4),
        "back_end_ratio": round(back_end, 4),
        "combined_dti": round(combined, 4),
        "risk_classification": risk_class,
        "risk_label": risk_label,
        "impact_description": impact,
        "thresholds": DTI_THRESHOLDS,
    }
