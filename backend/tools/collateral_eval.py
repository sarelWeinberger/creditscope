"""Collateral valuation tool."""


def evaluate_collateral(
    collateral_type: str, collateral_value: float, loan_amount: float
) -> dict:
    """
    Evaluate collateral adequacy for a loan.

    Returns LTV ratio, coverage ratio, risk assessment, and score adjustment.
    """
    if loan_amount <= 0:
        return {"error": "Loan amount must be positive"}

    ltv_ratio = loan_amount / collateral_value if collateral_value > 0 else float("inf")
    coverage_ratio = collateral_value / loan_amount if loan_amount > 0 else 0

    # Depreciation/volatility discount by type
    depreciation = {
        "real_estate": 0.95,   # Stable, minimal discount
        "vehicle": 0.80,       # Depreciates quickly
        "equipment": 0.70,     # Specialized, harder to liquidate
        "none": 0.0,
    }
    discount = depreciation.get(collateral_type, 0.50)
    adjusted_value = collateral_value * discount
    adjusted_ltv = loan_amount / adjusted_value if adjusted_value > 0 else float("inf")

    # Score adjustment
    if adjusted_ltv < 0.80:
        score_adj = int(10 + (0.80 - adjusted_ltv) / 0.80 * 10)
        risk = "low"
    elif adjusted_ltv <= 0.90:
        score_adj = 0
        risk = "moderate"
    elif adjusted_ltv <= 1.00:
        score_adj = -int(10 + (adjusted_ltv - 0.90) / 0.10 * 10)
        risk = "elevated"
    else:
        score_adj = -int(30 + min((adjusted_ltv - 1.0) / 0.20 * 20, 20))
        risk = "high"

    assessment_details = {
        "low": "Collateral provides strong coverage. Low risk of loss in default scenario.",
        "moderate": "Collateral provides adequate coverage. Standard risk profile.",
        "elevated": "Collateral coverage is thin. Consider requiring additional collateral or mortgage insurance.",
        "high": "Insufficient collateral coverage. Loan exceeds recoverable value in default scenario.",
    }

    return {
        "collateral_type": collateral_type,
        "collateral_value": collateral_value,
        "loan_amount": loan_amount,
        "ltv_ratio": round(ltv_ratio, 4),
        "adjusted_ltv_ratio": round(adjusted_ltv, 4),
        "coverage_ratio": round(coverage_ratio, 4),
        "depreciation_discount": discount,
        "adjusted_collateral_value": round(adjusted_value, 2),
        "risk_assessment": risk,
        "risk_details": assessment_details[risk],
        "score_adjustment": score_adj,
    }
