"""
Collateral evaluation tool.
Computes LTV ratio, coverage, haircut, and risk assessment per collateral type.
"""
from typing import Any, Dict


# Haircut percentages by collateral type (liquidity/depreciation discount)
HAIRCUT_RATES = {
    "real_estate": 0.10,       # 10% — illiquid but stable
    "vehicle": 0.20,           # 20% — depreciates rapidly
    "equipment": 0.30,         # 30% — specialized / illiquid
    "financial_asset": 0.05,   # 5% — liquid assets
    "none": 1.00,              # No collateral
}

RISK_THRESHOLDS = {
    "real_estate": {
        "low": 0.60,
        "medium": 0.80,
        "high": 0.95,
    },
    "vehicle": {
        "low": 0.70,
        "medium": 0.90,
        "high": 1.00,
    },
    "equipment": {
        "low": 0.65,
        "medium": 0.85,
        "high": 1.00,
    },
    "financial_asset": {
        "low": 0.65,
        "medium": 0.80,
        "high": 0.90,
    },
    "none": {
        "low": 0.0,
        "medium": 0.0,
        "high": 0.0,
    },
}


def _assess_risk(collateral_type: str, ltv: float) -> tuple[str, str, float]:
    """
    Assess collateral risk based on LTV ratio.
    Returns (risk_level, description, score_impact).
    """
    if collateral_type == "none":
        return "very_high", "No collateral — unsecured lending at full risk exposure.", -10.0

    thresholds = RISK_THRESHOLDS.get(collateral_type, RISK_THRESHOLDS["financial_asset"])

    if ltv <= thresholds["low"]:
        risk = "low"
        description = (
            f"Excellent collateral coverage (LTV {ltv*100:.1f}%). "
            f"Significant equity cushion in the {collateral_type.replace('_',' ')}."
        )
        score_impact = +20.0
    elif ltv <= thresholds["medium"]:
        risk = "medium"
        description = (
            f"Adequate collateral coverage (LTV {ltv*100:.1f}%). "
            f"Standard risk for {collateral_type.replace('_',' ')} collateral."
        )
        score_impact = +10.0
    elif ltv <= thresholds["high"]:
        risk = "high"
        description = (
            f"Limited collateral coverage (LTV {ltv*100:.1f}%). "
            f"Minimal equity; elevated risk if collateral depreciates."
        )
        score_impact = 0.0
    else:
        risk = "very_high"
        description = (
            f"Insufficient collateral coverage (LTV {ltv*100:.1f}%). "
            f"Loan amount exceeds collateral value; high exposure."
        )
        score_impact = -10.0

    return risk, description, score_impact


def evaluate_collateral(
    collateral_type: str,
    collateral_value: float,
    loan_amount: float,
) -> Dict[str, Any]:
    """
    Evaluate collateral for a loan application.

    Args:
        collateral_type: Type of collateral (real_estate/vehicle/equipment/financial_asset/none)
        collateral_value: Current market value of collateral in dollars
        loan_amount: Requested loan amount in dollars

    Returns:
        dict matching CollateralResponse schema
    """
    collateral_type = collateral_type.lower().replace(" ", "_")
    haircut = HAIRCUT_RATES.get(collateral_type, 0.20)

    if collateral_value <= 0 or collateral_type == "none":
        return {
            "collateral_type": collateral_type,
            "collateral_value": 0.0,
            "loan_amount": loan_amount,
            "ltv_ratio": 999.0,
            "coverage_ratio": 0.0,
            "haircut_pct": 1.0,
            "adjusted_value": 0.0,
            "risk_assessment": "very_high",
            "risk_description": "No collateral provided. Unsecured loan.",
            "score_impact": -10.0,
        }

    ltv = loan_amount / collateral_value
    coverage = collateral_value / loan_amount if loan_amount > 0 else 0.0
    adjusted_value = collateral_value * (1 - haircut)

    risk, description, score_impact = _assess_risk(collateral_type, ltv)

    return {
        "collateral_type": collateral_type,
        "collateral_value": round(collateral_value, 2),
        "loan_amount": round(loan_amount, 2),
        "ltv_ratio": round(ltv, 4),
        "coverage_ratio": round(coverage, 4),
        "haircut_pct": round(haircut, 4),
        "adjusted_value": round(adjusted_value, 2),
        "risk_assessment": risk,
        "risk_description": description,
        "score_impact": score_impact,
    }
