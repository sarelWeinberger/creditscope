"""
Loan-adjusted credit scoring tool.

Adjusts the base credit score based on a proposed loan structure,
factoring in DTI impact, collateral coverage, term risk, and purpose weight.
"""

from backend.tools.credit_score import calculate_base_credit_score


def calculate_loan_adjusted_score(
    customer,
    loan_amount: float,
    loan_term_months: int,
    interest_rate: float,
    loan_type: str,
    collateral_type: str = "none",
    collateral_value: float = 0,
) -> dict:
    """Calculate loan-adjusted credit score."""
    base_result = calculate_base_credit_score(customer)
    base_score = base_result["score"]
    adjustments = {}
    risk_factors = list(base_result["factors"])

    # 1. DTI Impact
    monthly_rate = interest_rate / 100 / 12
    if monthly_rate > 0:
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term_months) / (
            (1 + monthly_rate) ** loan_term_months - 1
        )
    else:
        monthly_payment = loan_amount / loan_term_months

    gross_monthly_income = customer.annual_income / 12
    current_dti = customer.monthly_debt_payments / gross_monthly_income if gross_monthly_income > 0 else 1.0
    projected_dti = (customer.monthly_debt_payments + monthly_payment) / gross_monthly_income if gross_monthly_income > 0 else 1.0

    dti_adjustment = _dti_adjustment(projected_dti)
    adjustments["dti_impact"] = dti_adjustment

    if projected_dti > 0.43:
        risk_factors.append(f"High projected DTI ratio: {projected_dti:.1%}")
    elif projected_dti > 0.36:
        risk_factors.append(f"Elevated projected DTI ratio: {projected_dti:.1%}")

    # 2. Collateral Coverage
    ltv_ratio = None
    collateral_adjustment = 0
    if collateral_type != "none" and collateral_value > 0:
        ltv_ratio = loan_amount / collateral_value
        collateral_adjustment = _collateral_adjustment(ltv_ratio)
        adjustments["collateral_coverage"] = collateral_adjustment

        if ltv_ratio > 1.0:
            risk_factors.append(f"Loan exceeds collateral value (LTV: {ltv_ratio:.1%})")
        elif ltv_ratio < 0.8:
            risk_factors.append(f"Strong collateral coverage (LTV: {ltv_ratio:.1%})")
    else:
        risk_factors.append("No collateral — unsecured loan")
        adjustments["collateral_coverage"] = -15
        collateral_adjustment = -15

    # 3. Term Risk
    term_adjustment = _term_adjustment(loan_term_months)
    adjustments["term_risk"] = term_adjustment

    # 4. Loan Purpose Weight
    purpose_multiplier = _purpose_weight(loan_type)
    adjustments["purpose_weight_multiplier"] = purpose_multiplier

    # Calculate adjusted score
    total_adjustment = dti_adjustment + collateral_adjustment + term_adjustment
    # Apply purpose multiplier to the penalty portion
    if total_adjustment < 0:
        total_adjustment = int(total_adjustment * purpose_multiplier)

    adjusted_score = max(300, min(850, base_score + total_adjustment))
    grade = _adjusted_grade(adjusted_score)

    recommendation = _generate_recommendation(adjusted_score, projected_dti, ltv_ratio, loan_type)

    return {
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "grade": grade,
        "adjustments": adjustments,
        "dti_ratio": round(projected_dti, 4),
        "ltv_ratio": round(ltv_ratio, 4) if ltv_ratio is not None else None,
        "monthly_payment": round(monthly_payment, 2),
        "risk_factors": risk_factors,
        "recommendation": recommendation,
    }


def _dti_adjustment(dti: float) -> int:
    """DTI-based score adjustment."""
    if dti < 0.28:
        return 0
    if dti < 0.36:
        return -int(10 + (dti - 0.28) / 0.08 * 20)
    if dti < 0.43:
        return -int(30 + (dti - 0.36) / 0.07 * 30)
    return -int(60 + min((dti - 0.43) / 0.17 * 60, 60))


def _collateral_adjustment(ltv: float) -> int:
    """Collateral LTV-based score adjustment."""
    if ltv < 0.80:
        return int(10 + (0.80 - ltv) / 0.80 * 10)
    if ltv <= 0.90:
        return 0
    if ltv <= 1.00:
        return -int(10 + (ltv - 0.90) / 0.10 * 10)
    return -int(30 + min((ltv - 1.0) / 0.20 * 20, 20))


def _term_adjustment(term_months: int) -> int:
    """Term length risk adjustment."""
    if term_months < 36:
        return 5
    if term_months <= 60:
        return 0
    if term_months <= 120:
        return -5
    return -15


def _purpose_weight(loan_type: str) -> float:
    """Loan purpose risk weight multiplier."""
    weights = {
        "mortgage": 1.0,
        "auto": 1.1,
        "student": 0.9,
        "personal": 1.3,
        "business": 1.4,
    }
    return weights.get(loan_type, 1.0)


def _adjusted_grade(score: int) -> str:
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


def _generate_recommendation(score: int, dti: float, ltv: float | None, loan_type: str) -> str:
    """Generate a human-readable recommendation."""
    if score >= 750:
        return "Excellent credit profile. Recommend automatic approval with standard terms."
    if score >= 700:
        if dti > 0.36:
            return "Good credit score but elevated DTI. Recommend approval with income verification conditions."
        return "Good credit profile. Recommend standard approval."
    if score >= 650:
        conditions = []
        if dti > 0.36:
            conditions.append("DTI reduction plan")
        if ltv and ltv > 0.90:
            conditions.append("additional collateral or down payment")
        cond_str = " and ".join(conditions) if conditions else "enhanced monitoring"
        return f"Fair credit profile. Recommend conditional approval with {cond_str}."
    if score >= 600:
        return "Below average credit profile. Recommend senior review before decision."
    if score >= 550:
        return "Poor credit profile. Likely decline unless strong compensating factors exist."
    return "Very poor credit profile. Recommend decline."
