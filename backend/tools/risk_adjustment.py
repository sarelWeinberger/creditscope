"""
Combined risk weighting tool.

Final composite scoring that combines all tool outputs into a risk grade:
- A  (750-850): Excellent — automatic approval recommended
- B  (700-749): Good — standard approval
- C  (650-699): Fair — approval with conditions
- D  (600-649): Below Average — requires senior review
- E  (550-599): Poor — likely decline
- F  (300-549): Very Poor — decline
"""

from backend.tools.credit_score import calculate_base_credit_score
from backend.tools.debt_to_income import calculate_dti_ratio
from backend.tools.payment_history import analyze_payment_history


def compute_risk_weighted_score(customer, loan_application=None) -> dict:
    """
    Compute final composite risk score combining all factors.

    Args:
        customer: Customer ORM object
        loan_application: Optional LoanApplication ORM object

    Returns:
        Composite risk assessment
    """
    # Component scores
    credit_result = calculate_base_credit_score(customer)
    dti_result = calculate_dti_ratio(customer)
    payment_result = analyze_payment_history(customer)

    base_score = credit_result["score"]
    payment_severity = payment_result["severity_score"]
    current_dti = dti_result.get("current_dti") or 0

    # Normalize components to 0-100
    credit_component = (base_score - 300) / 550 * 100  # 0-100
    payment_component = 100 - payment_severity            # 0-100 (inverted)
    dti_component = max(0, 100 - current_dti * 200)       # 0-100

    # If loan application provided, factor in loan-specific risk
    loan_component = 50  # neutral default
    if loan_application:
        loan_component = _score_loan_application(customer, loan_application)

    # Weighted composite
    weights = {
        "credit_score": 0.35,
        "payment_history": 0.25,
        "dti": 0.20,
        "loan_specific": 0.20,
    }

    composite = (
        credit_component * weights["credit_score"]
        + payment_component * weights["payment_history"]
        + dti_component * weights["dti"]
        + loan_component * weights["loan_specific"]
    )

    # Map back to 300-850
    final_score = int(300 + (composite / 100) * 550)
    final_score = max(300, min(850, final_score))

    grade = _score_to_grade(final_score)
    confidence = _calculate_confidence(customer)
    risk_factors = _compile_risk_factors(credit_result, dti_result, payment_result, loan_application)
    recommendation = _generate_recommendation(grade, risk_factors)

    return {
        "composite_score": final_score,
        "risk_grade": grade,
        "component_scores": {
            "credit_score": round(credit_component, 1),
            "payment_history": round(payment_component, 1),
            "debt_to_income": round(dti_component, 1),
            "loan_specific": round(loan_component, 1),
        },
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "confidence": round(confidence, 2),
        "base_credit_score": base_score,
        "dti_ratio": current_dti,
        "payment_severity": payment_severity,
    }


def _score_loan_application(customer, loan) -> float:
    """Score the loan application characteristics (0-100)."""
    score = 50.0  # Start at neutral

    # Amount relative to income
    if customer.annual_income > 0:
        income_ratio = loan.requested_amount / customer.annual_income
        if income_ratio < 1:
            score += 15
        elif income_ratio < 3:
            score += 5
        elif income_ratio > 10:
            score -= 20
        elif income_ratio > 5:
            score -= 10

    # Collateral
    if loan.collateral_value and loan.collateral_value > 0:
        ltv = loan.requested_amount / loan.collateral_value
        if ltv < 0.8:
            score += 15
        elif ltv > 1.0:
            score -= 15
    else:
        score -= 10  # Unsecured

    # Term risk
    if loan.proposed_term_months < 36:
        score += 5
    elif loan.proposed_term_months > 120:
        score -= 10

    # Loan type risk
    type_adjustments = {
        "student": 5,
        "mortgage": 0,
        "auto": -5,
        "personal": -10,
        "business": -15,
    }
    score += type_adjustments.get(loan.loan_type, 0)

    return max(0, min(100, score))


def _score_to_grade(score: int) -> str:
    if score >= 750:
        return "A"
    if score >= 700:
        return "B"
    if score >= 650:
        return "C"
    if score >= 600:
        return "D"
    if score >= 550:
        return "E"
    return "F"


def _calculate_confidence(customer) -> float:
    """Calculate confidence in the assessment (0-1)."""
    confidence = 0.5

    # More data = higher confidence
    if customer.credit_history_years >= 5:
        confidence += 0.15
    if customer.credit_history_years >= 10:
        confidence += 0.10

    if customer.num_open_accounts >= 3:
        confidence += 0.10

    if customer.annual_income > 0:
        confidence += 0.10

    # Risk notes suggest manual review needed
    if customer.risk_notes:
        confidence -= 0.10

    return max(0.1, min(1.0, confidence))


def _compile_risk_factors(credit_result, dti_result, payment_result, loan=None) -> list[str]:
    """Compile all risk factors from component analyses."""
    factors = list(credit_result.get("factors", []))

    dti_val = dti_result.get("current_dti")
    if dti_val and dti_val > 0.43:
        factors.append(f"DTI ratio exceeds conventional limit ({dti_val:.1%})")
    elif dti_val and dti_val > 0.36:
        factors.append(f"DTI ratio elevated ({dti_val:.1%})")

    if payment_result["risk_level"] in ("high", "severe"):
        factors.append(f"Payment history risk: {payment_result['risk_level']}")
    if payment_result["delinquency_trend"] == "worsening":
        factors.append("Worsening delinquency trend")

    if loan:
        if loan.collateral_type == "none" or not loan.collateral_value:
            factors.append("Unsecured loan — no collateral")

    return factors


def _generate_recommendation(grade: str, risk_factors: list[str]) -> str:
    """Generate recommendation based on grade and factors."""
    recommendations = {
        "A": "Excellent risk profile. Recommend automatic approval with best available terms.",
        "B": "Good risk profile. Recommend standard approval.",
        "C": "Fair risk profile. Recommend conditional approval — review risk factors before finalizing.",
        "D": "Below average risk profile. Requires senior underwriter review before decision.",
        "E": "Poor risk profile. Likely decline. Only proceed if strong compensating factors exist.",
        "F": "Very poor risk profile. Recommend decline.",
    }
    return recommendations.get(grade, "Unable to determine recommendation.")
