"""
FICO-like credit score calculator.
Implements the five standard credit score factors with FICO-approximate weights.
"""
import math
from typing import Any, Dict, List


def _payment_history_score(customer) -> tuple[float, str, Dict[str, Any]]:
    """
    Payment History (35% weight).
    Penalizes late payments, collections, bankruptcies, foreclosures, charge-offs.
    """
    total_payments = customer.on_time_payments + customer.late_payments_30d + \
                     customer.late_payments_60d + customer.late_payments_90d

    if total_payments == 0:
        raw_score = 50.0  # No history
        explanation = "No payment history available."
        on_time_rate = 0.0
    else:
        # Weighted severity of delinquencies
        severity_penalty = (
            customer.late_payments_30d * 1.0
            + customer.late_payments_60d * 2.5
            + customer.late_payments_90d * 4.0
            + customer.collections * 8.0
            + customer.bankruptcies * 20.0
            + customer.foreclosures * 15.0
            + customer.charge_offs * 10.0
        )

        on_time_rate = customer.on_time_payments / total_payments
        base_score = on_time_rate * 100.0
        raw_score = max(0.0, base_score - severity_penalty * 2.0)
        raw_score = min(100.0, raw_score)

        if raw_score >= 90:
            explanation = "Excellent payment record with minimal or no delinquencies."
        elif raw_score >= 70:
            explanation = "Good payment history with few minor late payments."
        elif raw_score >= 50:
            explanation = "Mixed payment history with some delinquencies noted."
        elif raw_score >= 30:
            explanation = "Poor payment history with significant delinquencies."
        else:
            explanation = "Very poor payment history; multiple serious delinquencies, collections, or bankruptcies."

    breakdown = {
        "on_time_payments": customer.on_time_payments,
        "late_30d": customer.late_payments_30d,
        "late_60d": customer.late_payments_60d,
        "late_90d": customer.late_payments_90d,
        "collections": customer.collections,
        "bankruptcies": customer.bankruptcies,
        "foreclosures": customer.foreclosures,
        "charge_offs": customer.charge_offs,
        "on_time_rate": round(on_time_rate, 4) if total_payments > 0 else 0.0,
        "factor_score": round(raw_score, 2),
    }
    return raw_score, explanation, breakdown


def _utilization_score(customer) -> tuple[float, str, Dict[str, Any]]:
    """
    Credit Utilization (30% weight).
    Lower utilization = better score.
    """
    if customer.total_credit_limit <= 0:
        utilization = 0.0
        raw_score = 50.0
        explanation = "No revolving credit accounts found."
    else:
        utilization = customer.total_credit_used / customer.total_credit_limit
        utilization = min(utilization, 1.0)

        # Score function: 100 at 0%, rapid decay above 30%
        if utilization <= 0.10:
            raw_score = 95.0 + (0.10 - utilization) / 0.10 * 5.0
        elif utilization <= 0.30:
            raw_score = 80.0 + (0.30 - utilization) / 0.20 * 15.0
        elif utilization <= 0.50:
            raw_score = 55.0 + (0.50 - utilization) / 0.20 * 25.0
        elif utilization <= 0.75:
            raw_score = 25.0 + (0.75 - utilization) / 0.25 * 30.0
        else:
            raw_score = max(0.0, 25.0 * (1.0 - utilization))

        raw_score = min(100.0, max(0.0, raw_score))

        pct = round(utilization * 100, 1)
        if utilization <= 0.10:
            explanation = f"Excellent utilization at {pct}%. Keeping balances very low is optimal."
        elif utilization <= 0.30:
            explanation = f"Good utilization at {pct}%. Below 30% is generally recommended."
        elif utilization <= 0.50:
            explanation = f"Moderate utilization at {pct}%. Reducing balances will improve this factor."
        elif utilization <= 0.75:
            explanation = f"High utilization at {pct}%. Significantly impacts score; reduce balances."
        else:
            explanation = f"Very high utilization at {pct}%. Near or at credit limits is a major negative signal."

    breakdown = {
        "total_credit_limit": customer.total_credit_limit,
        "total_credit_used": customer.total_credit_used,
        "utilization_ratio": round(utilization, 4) if customer.total_credit_limit > 0 else 0.0,
        "factor_score": round(raw_score, 2),
    }
    return raw_score, explanation, breakdown


def _credit_age_score(customer) -> tuple[float, str, Dict[str, Any]]:
    """
    Length of Credit History (15% weight).
    Rewards longer histories.
    """
    years = customer.credit_history_years

    if years <= 0:
        raw_score = 10.0
        explanation = "No established credit history."
    elif years <= 1:
        raw_score = 25.0
        explanation = "Very new credit history (under 1 year)."
    elif years <= 2:
        raw_score = 40.0
        explanation = "Short credit history (1-2 years)."
    elif years <= 5:
        raw_score = 60.0
        explanation = "Developing credit history (2-5 years)."
    elif years <= 10:
        raw_score = 80.0
        explanation = "Established credit history (5-10 years)."
    elif years <= 20:
        raw_score = 92.0
        explanation = "Long credit history (10-20 years). Very positive."
    else:
        raw_score = 100.0
        explanation = "Excellent credit history length (20+ years)."

    breakdown = {
        "credit_history_years": years,
        "num_accounts": customer.num_credit_accounts,
        "open_accounts": customer.num_open_accounts,
        "factor_score": round(raw_score, 2),
    }
    return raw_score, explanation, breakdown


def _credit_mix_score(customer) -> tuple[float, str, Dict[str, Any]]:
    """
    Credit Mix (10% weight).
    Rewards having diverse types of credit.
    """
    account_types = 0
    type_details = []

    if customer.mortgage_balance > 0 or customer.property_value > 0:
        account_types += 1
        type_details.append("mortgage")
    if customer.auto_loan_balance > 0 or customer.vehicle_value > 0:
        account_types += 1
        type_details.append("auto loan")
    if customer.student_loan_balance > 0:
        account_types += 1
        type_details.append("student loan")
    if customer.credit_card_balance > 0 or customer.total_credit_limit > 0:
        account_types += 1
        type_details.append("revolving credit")
    if customer.other_debt > 0:
        account_types += 1
        type_details.append("other installment")

    if account_types == 0:
        raw_score = 20.0
        explanation = "No identifiable credit mix."
    elif account_types == 1:
        raw_score = 45.0
        explanation = f"Limited mix: only {type_details[0]}."
    elif account_types == 2:
        raw_score = 65.0
        explanation = f"Moderate mix: {' and '.join(type_details)}."
    elif account_types == 3:
        raw_score = 82.0
        explanation = f"Good mix: {', '.join(type_details)}."
    else:
        raw_score = 95.0
        explanation = f"Excellent mix: {', '.join(type_details)}."

    breakdown = {
        "account_types_count": account_types,
        "account_types": type_details,
        "factor_score": round(raw_score, 2),
    }
    return raw_score, explanation, breakdown


def _new_credit_score(customer) -> tuple[float, str, Dict[str, Any]]:
    """
    New Credit / Inquiries (10% weight).
    Penalizes recent hard inquiries.
    """
    inquiries_6m = customer.hard_inquiries_6m
    inquiries_12m = customer.hard_inquiries_12m

    # 6-month inquiries are more penalizing
    penalty = inquiries_6m * 8.0 + max(0, inquiries_12m - inquiries_6m) * 4.0
    raw_score = max(0.0, 100.0 - penalty)

    if inquiries_6m == 0 and inquiries_12m == 0:
        explanation = "No recent hard inquiries. Optimal."
    elif inquiries_6m <= 1:
        explanation = f"Minimal recent inquiries ({inquiries_6m} in 6 months). Minor impact."
    elif inquiries_6m <= 3:
        explanation = f"Moderate inquiry activity ({inquiries_6m} in 6 months). Some negative impact."
    else:
        explanation = f"High inquiry activity ({inquiries_6m} in 6 months, {inquiries_12m} in 12 months). Significant negative impact."

    breakdown = {
        "hard_inquiries_6m": inquiries_6m,
        "hard_inquiries_12m": inquiries_12m,
        "penalty_points": round(penalty, 2),
        "factor_score": round(raw_score, 2),
    }
    return raw_score, explanation, breakdown


def _score_to_grade(score: int) -> tuple[str, str]:
    """Convert numeric score to letter grade and label."""
    if score >= 750:
        return "A", "Excellent"
    elif score >= 700:
        return "B", "Good"
    elif score >= 650:
        return "C", "Fair"
    elif score >= 600:
        return "D", "Poor"
    elif score >= 550:
        return "E", "Very Poor"
    return "F", "Extremely Poor"


def _generate_recommendations(customer, breakdown: Dict[str, Any]) -> List[str]:
    """Generate actionable improvement recommendations."""
    recs = []

    util = breakdown.get("utilization", {}).get("utilization_ratio", 0)
    if util > 0.30:
        recs.append(
            f"Reduce credit card balances. Current utilization ({util*100:.0f}%) exceeds the "
            "recommended 30% threshold."
        )

    if customer.late_payments_30d + customer.late_payments_60d + customer.late_payments_90d > 0:
        recs.append("Set up automatic payments to avoid future late payments.")

    if customer.credit_history_years < 3:
        recs.append(
            "Build credit history over time. Avoid closing old accounts to maintain account age."
        )

    if customer.hard_inquiries_6m > 2:
        recs.append("Limit new credit applications. Multiple inquiries in a short period lower your score.")

    if customer.num_open_accounts < 2:
        recs.append(
            "Consider diversifying credit types (e.g., a small installment loan alongside revolving credit)."
        )

    if customer.collections > 0:
        recs.append(
            "Address outstanding collection accounts — negotiate pay-for-delete or settle accounts."
        )

    if not recs:
        recs.append("Your credit profile is strong. Continue maintaining current habits.")

    return recs


def calculate_base_credit_score(customer) -> Dict[str, Any]:
    """
    Calculate a FICO-like credit score for a customer.

    Weights:
      - Payment History: 35%
      - Credit Utilization: 30%
      - Credit History Length: 15%
      - Credit Mix: 10%
      - New Credit: 10%

    Returns a dict matching CreditScoreResponse schema.
    """
    ph_score, ph_expl, ph_breakdown = _payment_history_score(customer)
    util_score, util_expl, util_breakdown = _utilization_score(customer)
    age_score, age_expl, age_breakdown = _credit_age_score(customer)
    mix_score, mix_expl, mix_breakdown = _credit_mix_score(customer)
    new_score, new_expl, new_breakdown = _new_credit_score(customer)

    # Weighted composite (0-100)
    weighted = (
        ph_score * 0.35
        + util_score * 0.30
        + age_score * 0.15
        + mix_score * 0.10
        + new_score * 0.10
    )

    # Scale to 300-850
    score = int(round(300 + weighted / 100 * 550))
    score = max(300, min(850, score))

    grade, grade_label = _score_to_grade(score)

    breakdown = {
        "payment_history": ph_breakdown,
        "utilization": util_breakdown,
        "credit_age": age_breakdown,
        "credit_mix": mix_breakdown,
        "new_credit": new_breakdown,
        "weighted_composite": round(weighted, 2),
    }

    recommendations = _generate_recommendations(customer, breakdown)

    return {
        "customer_id": customer.id,
        "score": score,
        "grade": grade,
        "grade_label": grade_label,
        "payment_history_score": round(ph_score, 2),
        "credit_utilization_score": round(util_score, 2),
        "credit_age_score": round(age_score, 2),
        "credit_mix_score": round(mix_score, 2),
        "new_credit_score": round(new_score, 2),
        "payment_history_weight": 35.0,
        "credit_utilization_weight": 30.0,
        "credit_age_weight": 15.0,
        "credit_mix_weight": 10.0,
        "new_credit_weight": 10.0,
        "factors": {
            "payment_history": ph_expl,
            "credit_utilization": util_expl,
            "credit_history_length": age_expl,
            "credit_mix": mix_expl,
            "new_credit": new_expl,
        },
        "breakdown": breakdown,
        "recommendations": recommendations,
    }
