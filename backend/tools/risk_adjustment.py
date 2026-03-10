"""
Risk-weighted composite credit scoring.
Combines all tool outputs into a final risk assessment.
"""
from typing import Any, Dict, List, Optional

from tools.credit_score import calculate_base_credit_score, _score_to_grade
from tools.debt_to_income import calculate_dti_ratio
from tools.payment_history import analyze_payment_history


RISK_GRADE_THRESHOLDS = {
    "A": 750,
    "B": 700,
    "C": 650,
    "D": 600,
    "E": 550,
    "F": 0,
}

GRADE_LABELS = {
    "A": "Prime — Excellent creditworthiness",
    "B": "Near Prime — Good credit standing",
    "C": "Subprime — Fair credit; elevated risk",
    "D": "Deep Subprime — Poor credit; high risk",
    "E": "Very High Risk — Significant concerns",
    "F": "Extremely High Risk — Decline recommended",
}

RECOMMENDATIONS = {
    "A": "Approve — Low risk. Standard rate terms recommended.",
    "B": "Approve — Acceptable risk. Minor rate premium may apply.",
    "C": "Conditional Approval — Review compensating factors. Higher rate tier.",
    "D": "Review Required — Significant risk. Collateral or co-signer recommended.",
    "E": "Decline or Hold — Risk exceeds standard thresholds. Exceptional circumstances only.",
    "F": "Decline — Risk profile does not meet lending criteria.",
}


def _grade_from_score(score: float) -> str:
    """Return grade letter for a composite risk score."""
    for grade, threshold in RISK_GRADE_THRESHOLDS.items():
        if score >= threshold:
            return grade
    return "F"


def _identify_risk_factors(customer, base_result: Dict, dti_result: Dict, payment_result: Dict) -> List[str]:
    """Identify the top negative risk factors."""
    factors = []

    # Payment history
    if customer.bankruptcies > 0:
        factors.append(f"Bankruptcy on record ({customer.bankruptcies} filing(s))")
    if customer.foreclosures > 0:
        factors.append(f"Foreclosure on record ({customer.foreclosures} event(s))")
    if customer.collections > 0:
        factors.append(f"Active collection accounts ({customer.collections})")
    if customer.charge_offs > 0:
        factors.append(f"Charged-off accounts ({customer.charge_offs})")
    if customer.late_payments_90d > 0:
        factors.append(f"90-day late payments ({customer.late_payments_90d})")
    if customer.late_payments_60d > 2:
        factors.append(f"Multiple 60-day late payments ({customer.late_payments_60d})")

    # Utilization
    util = customer.total_credit_used / max(customer.total_credit_limit, 1)
    if util > 0.75:
        factors.append(f"Very high credit utilization ({util*100:.0f}%)")
    elif util > 0.50:
        factors.append(f"High credit utilization ({util*100:.0f}%)")

    # DTI
    combined_dti = dti_result.get("combined_dti", 0)
    if combined_dti > 0.50:
        factors.append(f"Excessive debt-to-income ratio ({combined_dti*100:.1f}%)")
    elif combined_dti > 0.43:
        factors.append(f"High debt-to-income ratio ({combined_dti*100:.1f}%)")

    # Credit history
    if customer.credit_history_years < 1:
        factors.append("Very limited credit history (< 1 year)")

    # Inquiries
    if customer.hard_inquiries_6m > 4:
        factors.append(f"High inquiry volume ({customer.hard_inquiries_6m} in 6 months)")

    return factors[:5]  # Top 5


def _identify_positive_factors(customer, base_result: Dict) -> List[str]:
    """Identify positive credit factors."""
    positives = []

    if customer.on_time_payments > 50 and customer.late_payments_30d == 0:
        positives.append(f"Strong on-time payment record ({customer.on_time_payments} payments)")

    util = customer.total_credit_used / max(customer.total_credit_limit, 1)
    if util <= 0.20:
        positives.append(f"Excellent credit utilization ({util*100:.0f}%)")

    if customer.credit_history_years >= 10:
        positives.append(f"Long credit history ({customer.credit_history_years:.0f} years)")

    if customer.savings_balance + customer.investment_balance > customer.annual_income:
        positives.append("Strong savings and investment reserves")

    monthly_income = customer.monthly_income or 1
    dti = customer.monthly_debt_payments / monthly_income
    if dti <= 0.28:
        positives.append(f"Low current debt-to-income ratio ({dti*100:.1f}%)")

    if customer.num_credit_accounts >= 5 and customer.bankruptcies == 0:
        positives.append(f"Diverse credit portfolio ({customer.num_credit_accounts} accounts)")

    if customer.hard_inquiries_6m == 0:
        positives.append("No recent hard inquiries")

    return positives[:5]  # Top 5


def compute_risk_weighted_score(customer, loan_application=None) -> Dict[str, Any]:
    """
    Compute a comprehensive risk-weighted credit score.

    Combines:
      - Base FICO-like score (primary)
      - DTI ratio assessment
      - Payment history analysis
      - Loan-specific adjustments (if loan_application provided)

    Risk grade thresholds:
      A: 750+, B: 700-749, C: 650-699, D: 600-649, E: 550-599, F: <550

    Returns:
        dict matching RiskWeightedResponse schema
    """
    base_result = calculate_base_credit_score(customer)
    dti_result = calculate_dti_ratio(customer)
    payment_result = analyze_payment_history(customer)

    base_score = base_result["score"]

    # Factor contributions (as points)
    ph_contribution = base_result["payment_history_score"] * 0.35 / 100 * 550
    util_contribution = base_result["credit_utilization_score"] * 0.30 / 100 * 550
    age_contribution = base_result["credit_age_score"] * 0.15 / 100 * 550
    mix_contribution = base_result["credit_mix_score"] * 0.10 / 100 * 550
    new_contribution = base_result["new_credit_score"] * 0.10 / 100 * 550

    # Penalty adjustments
    dti_penalty = 0.0
    combined_dti = dti_result["combined_dti"]
    if combined_dti > 0.50:
        dti_penalty = -30.0
    elif combined_dti > 0.43:
        dti_penalty = -15.0
    elif combined_dti > 0.36:
        dti_penalty = -5.0

    # Severity penalty from payment history
    severity_penalty = min(50.0, payment_result["severity_score"] * 0.5)

    risk_score = max(300.0, min(850.0, base_score + dti_penalty - severity_penalty * 0.3))

    risk_grade = _grade_from_score(risk_score)
    grade_label = GRADE_LABELS[risk_grade]
    recommendation = RECOMMENDATIONS[risk_grade]

    key_risks = _identify_risk_factors(customer, base_result, dti_result, payment_result)
    positives = _identify_positive_factors(customer, base_result)

    breakdown = {
        "base_score": base_score,
        "risk_score": round(risk_score, 1),
        "dti_penalty": round(dti_penalty, 2),
        "severity_penalty": round(severity_penalty * 0.3, 2),
        "combined_dti": dti_result["combined_dti"],
        "payment_on_time_rate": payment_result["on_time_rate"],
        "severity_score": payment_result["severity_score"],
        "delinquency_trend": payment_result["trend"],
        "factor_scores": {
            "payment_history": base_result["payment_history_score"],
            "credit_utilization": base_result["credit_utilization_score"],
            "credit_age": base_result["credit_age_score"],
            "credit_mix": base_result["credit_mix_score"],
            "new_credit": base_result["new_credit_score"],
        },
    }

    return {
        "customer_id": customer.id,
        "base_score": base_score,
        "payment_history_contribution": round(ph_contribution, 2),
        "utilization_contribution": round(util_contribution, 2),
        "credit_age_contribution": round(age_contribution, 2),
        "credit_mix_contribution": round(mix_contribution, 2),
        "new_credit_contribution": round(new_contribution, 2),
        "risk_score": round(risk_score, 1),
        "risk_grade": risk_grade,
        "risk_label": grade_label,
        "recommendation": recommendation,
        "breakdown": breakdown,
        "key_risk_factors": key_risks,
        "positive_factors": positives,
    }
