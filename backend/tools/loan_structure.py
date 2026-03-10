"""
Loan-adjusted credit scoring.
Adjusts base credit score based on loan-specific factors.
"""
from typing import Any, Dict

from tools.credit_score import calculate_base_credit_score, _score_to_grade


# Loan purpose risk multipliers (>1.0 = higher risk)
LOAN_PURPOSE_WEIGHTS = {
    "home_purchase": 0.95,
    "refinance": 0.97,
    "home_improvement": 0.98,
    "auto": 0.97,
    "education": 0.98,
    "debt_consolidation": 1.02,
    "medical": 1.05,
    "vacation": 1.15,
    "business": 1.05,
    "personal": 1.08,
    "other": 1.10,
}

# Term risk: longer terms carry more risk
TERM_RISK_ADJUSTMENTS = {
    12: +5,
    24: +3,
    36: +1,
    48: 0,
    60: -2,
    72: -4,
    84: -6,
    120: -8,
    180: -10,
    240: -12,
    360: -15,
}


def _dti_adjustment(customer, loan_params) -> tuple[float, str]:
    """
    Calculate DTI impact on adjusted score.
    Returns (adjustment_points, explanation).
    """
    monthly_income = customer.monthly_income
    if monthly_income <= 0:
        return -50.0, "Unable to calculate DTI: no monthly income."

    # Compute new monthly payment if not provided
    new_payment = loan_params.get("additional_monthly_payment")
    if new_payment is None or new_payment == 0:
        rate = 0.06 / 12  # default 6% assumption
        term = loan_params.get("term_months", 60)
        amount = loan_params.get("loan_amount", 0)
        if rate > 0 and term > 0:
            new_payment = amount * rate / (1 - (1 + rate) ** (-term))
        else:
            new_payment = amount / max(term, 1)

    current_dti = customer.monthly_debt_payments / monthly_income
    new_dti = (customer.monthly_debt_payments + new_payment) / monthly_income

    if new_dti <= 0.28:
        adjustment = +10.0
        explanation = f"Post-loan DTI ({new_dti*100:.1f}%) is excellent. Positive impact."
    elif new_dti <= 0.36:
        adjustment = +5.0
        explanation = f"Post-loan DTI ({new_dti*100:.1f}%) is good."
    elif new_dti <= 0.43:
        adjustment = 0.0
        explanation = f"Post-loan DTI ({new_dti*100:.1f}%) is at the standard lending threshold."
    elif new_dti <= 0.50:
        adjustment = -15.0
        explanation = f"Post-loan DTI ({new_dti*100:.1f}%) exceeds recommended levels. Negative impact."
    elif new_dti <= 0.60:
        adjustment = -30.0
        explanation = f"Post-loan DTI ({new_dti*100:.1f}%) is high. Significant risk."
    else:
        adjustment = -50.0
        explanation = f"Post-loan DTI ({new_dti*100:.1f}%) is dangerously high. Severe negative impact."

    return adjustment, explanation


def _collateral_adjustment(loan_params) -> tuple[float, str]:
    """
    Calculate collateral coverage impact on adjusted score.
    Returns (adjustment_points, explanation).
    """
    collateral_type = loan_params.get("collateral_type", "none")
    collateral_value = loan_params.get("collateral_value", 0.0)
    loan_amount = loan_params.get("loan_amount", 1.0)

    if collateral_type == "none" or collateral_value <= 0:
        return -5.0, "No collateral provided. Unsecured loan; slight negative adjustment."

    ltv = loan_amount / collateral_value if collateral_value > 0 else 999

    if collateral_type == "real_estate":
        if ltv <= 0.60:
            return +20.0, f"Strong real estate collateral (LTV {ltv*100:.1f}%). Excellent coverage."
        elif ltv <= 0.80:
            return +12.0, f"Good real estate collateral (LTV {ltv*100:.1f}%)."
        elif ltv <= 0.95:
            return +5.0, f"Moderate real estate collateral (LTV {ltv*100:.1f}%)."
        else:
            return -5.0, f"High LTV on real estate ({ltv*100:.1f}%). Limited collateral benefit."
    elif collateral_type == "vehicle":
        if ltv <= 0.80:
            return +10.0, f"Good vehicle collateral (LTV {ltv*100:.1f}%)."
        elif ltv <= 1.0:
            return +3.0, f"Marginal vehicle collateral (LTV {ltv*100:.1f}%)."
        else:
            return -3.0, f"Vehicle loan exceeds collateral value (LTV {ltv*100:.1f}%)."
    elif collateral_type == "equipment":
        if ltv <= 0.70:
            return +8.0, f"Good equipment collateral (LTV {ltv*100:.1f}%)."
        elif ltv <= 0.90:
            return +3.0, f"Adequate equipment collateral (LTV {ltv*100:.1f}%)."
        else:
            return -2.0, f"Equipment collateral provides limited coverage (LTV {ltv*100:.1f}%)."
    elif collateral_type == "financial_asset":
        if ltv <= 0.70:
            return +15.0, f"Strong financial asset collateral (LTV {ltv*100:.1f}%)."
        else:
            return +5.0, f"Financial asset collateral (LTV {ltv*100:.1f}%)."
    else:
        return 0.0, "Other collateral type; neutral adjustment."


def _term_risk_adjustment(term_months: int) -> tuple[float, str]:
    """
    Calculate term risk adjustment.
    Shorter terms are less risky; longer terms add exposure.
    """
    # Find closest key
    keys = sorted(TERM_RISK_ADJUSTMENTS.keys())
    closest = min(keys, key=lambda k: abs(k - term_months))
    adjustment = float(TERM_RISK_ADJUSTMENTS[closest])

    years = term_months / 12
    if adjustment > 0:
        explanation = f"Short loan term ({years:.1f} years). Lower duration risk, positive adjustment."
    elif adjustment == 0:
        explanation = f"Standard loan term ({years:.1f} years). Neutral term risk."
    else:
        explanation = f"Long loan term ({years:.1f} years). Extended duration risk, negative adjustment."

    return adjustment, explanation


def _purpose_adjustment(loan_type: str, loan_purpose: str = "") -> tuple[float, str]:
    """
    Calculate loan purpose risk adjustment.
    """
    purpose_key = loan_type.lower()
    if purpose_key in LOAN_PURPOSE_WEIGHTS:
        weight = LOAN_PURPOSE_WEIGHTS[purpose_key]
    else:
        # Try to match from purpose text
        purpose_lower = loan_purpose.lower()
        if any(k in purpose_lower for k in ["home", "house", "property"]):
            weight = LOAN_PURPOSE_WEIGHTS["home_purchase"]
        elif any(k in purpose_lower for k in ["car", "vehicle", "auto"]):
            weight = LOAN_PURPOSE_WEIGHTS["auto"]
        elif any(k in purpose_lower for k in ["vacation", "travel", "holiday"]):
            weight = LOAN_PURPOSE_WEIGHTS["vacation"]
        elif any(k in purpose_lower for k in ["business", "company", "startup"]):
            weight = LOAN_PURPOSE_WEIGHTS["business"]
        elif any(k in purpose_lower for k in ["medical", "health", "hospital"]):
            weight = LOAN_PURPOSE_WEIGHTS["medical"]
        else:
            weight = LOAN_PURPOSE_WEIGHTS["other"]

    # Convert multiplier to score adjustment
    if weight <= 0.95:
        adjustment = +15.0
    elif weight <= 0.98:
        adjustment = +8.0
    elif weight <= 1.00:
        adjustment = +3.0
    elif weight <= 1.02:
        adjustment = 0.0
    elif weight <= 1.05:
        adjustment = -5.0
    elif weight <= 1.08:
        adjustment = -10.0
    else:
        adjustment = -20.0

    if adjustment >= 0:
        explanation = f"Loan purpose '{loan_type}' is a lower-risk category. Positive adjustment."
    else:
        explanation = f"Loan purpose '{loan_type}' carries higher risk profile. Negative adjustment."

    return adjustment, explanation


def calculate_loan_adjusted_score(customer, loan_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate a loan-adjusted credit score.

    Adjustments applied to the base score:
      1. DTI impact (post-loan debt-to-income ratio)
      2. Collateral coverage (LTV-based)
      3. Term risk (loan duration)
      4. Purpose weight (loan type risk multiplier)

    Returns a dict matching LoanAdjustedScoreResponse schema.
    """
    base_result = calculate_base_credit_score(customer)
    base_score = base_result["score"]

    dti_adj, dti_expl = _dti_adjustment(customer, loan_params)
    coll_adj, coll_expl = _collateral_adjustment(loan_params)
    term_adj, term_expl = _term_risk_adjustment(loan_params.get("term_months", 60))
    purpose_adj, purpose_expl = _purpose_adjustment(
        loan_params.get("loan_type", "personal"),
        loan_params.get("loan_purpose", ""),
    )

    total_adjustment = dti_adj + coll_adj + term_adj + purpose_adj
    adjusted_score = int(max(300, min(850, base_score + total_adjustment)))

    grade, grade_label = _score_to_grade(adjusted_score)

    # Recommendation
    if adjusted_score >= 720 and customer.monthly_debt_payments / max(customer.monthly_income, 1) <= 0.43:
        recommendation = "approve"
    elif adjusted_score >= 640:
        recommendation = "conditional"
    else:
        recommendation = "deny"

    return {
        "customer_id": customer.id,
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "final_grade": grade,
        "final_grade_label": grade_label,
        "dti_adjustment": round(dti_adj, 2),
        "collateral_adjustment": round(coll_adj, 2),
        "term_risk_adjustment": round(term_adj, 2),
        "purpose_adjustment": round(purpose_adj, 2),
        "total_adjustment": round(total_adjustment, 2),
        "adjustments": {
            "dti_impact": dti_expl,
            "collateral_coverage": coll_expl,
            "term_risk": term_expl,
            "loan_purpose": purpose_expl,
        },
        "loan_recommendation": recommendation,
    }
