"""
Payment history analysis tool.
Analyzes delinquency severity, recovery patterns, and trend indicators.
"""
from typing import Any, Dict


# Severity weights for different delinquency types
SEVERITY_WEIGHTS = {
    "late_30d": 1.0,
    "late_60d": 2.5,
    "late_90d": 4.0,
    "collections": 8.0,
    "charge_offs": 10.0,
    "foreclosures": 15.0,
    "bankruptcies": 20.0,
}


def _compute_trend(customer) -> tuple[str, str]:
    """
    Estimate payment trend based on ratio of recent vs older delinquencies.
    Uses on-time count vs total delinquencies as a proxy.
    """
    total_delinquencies = (
        customer.late_payments_30d
        + customer.late_payments_60d
        + customer.late_payments_90d
        + customer.collections
        + customer.bankruptcies
        + customer.foreclosures
        + customer.charge_offs
    )
    total_payments = customer.on_time_payments + total_delinquencies

    if total_payments == 0:
        return "stable", "No payment history to determine trend."

    on_time_rate = customer.on_time_payments / total_payments

    # Severe marks indicate past crisis
    severe_marks = customer.bankruptcies + customer.foreclosures + customer.charge_offs

    if severe_marks > 0 and on_time_rate > 0.85:
        return "improving", "Past severe delinquencies with recent improvement in on-time payments."
    elif on_time_rate >= 0.95 and total_delinquencies <= 1:
        return "improving", "Consistently strong payment history. Excellent trend."
    elif on_time_rate >= 0.90:
        return "stable", "Predominantly on-time payments with minimal delinquencies."
    elif on_time_rate >= 0.75:
        return "stable", "Mixed payment history; room for improvement."
    elif total_delinquencies > 5:
        return "deteriorating", "Multiple delinquencies indicate a deteriorating payment pattern."
    else:
        return "deteriorating", "Significant delinquencies observed; pattern is concerning."


def _severity_score(customer) -> float:
    """
    Calculate a weighted severity score for delinquencies.
    Higher = worse. Normalized to 0-100 scale.
    """
    raw = (
        customer.late_payments_30d * SEVERITY_WEIGHTS["late_30d"]
        + customer.late_payments_60d * SEVERITY_WEIGHTS["late_60d"]
        + customer.late_payments_90d * SEVERITY_WEIGHTS["late_90d"]
        + customer.collections * SEVERITY_WEIGHTS["collections"]
        + customer.charge_offs * SEVERITY_WEIGHTS["charge_offs"]
        + customer.foreclosures * SEVERITY_WEIGHTS["foreclosures"]
        + customer.bankruptcies * SEVERITY_WEIGHTS["bankruptcies"]
    )
    # Cap at 100 for normalization
    return min(100.0, raw)


def analyze_payment_history(customer) -> Dict[str, Any]:
    """
    Analyze a customer's payment history for credit risk assessment.

    Factors:
      - On-time rate (primary)
      - Delinquency severity (weighted by days past due)
      - Recovery pattern (trend analysis)
      - Serious derogatory marks (collections, bankruptcies, etc.)

    Returns:
        dict matching PaymentHistoryResponse schema
    """
    total_late = (
        customer.late_payments_30d
        + customer.late_payments_60d
        + customer.late_payments_90d
    )
    total_payments = customer.on_time_payments + total_late

    on_time_rate = (
        customer.on_time_payments / total_payments if total_payments > 0 else 0.0
    )

    # Delinquency score (0-100, higher = more delinquent)
    if total_payments == 0:
        delinquency_score = 50.0  # Unknown
    else:
        delinquency_rate = total_late / total_payments
        delinquency_score = min(100.0, delinquency_rate * 100 * 2)

    severity = _severity_score(customer)
    trend, trend_description = _compute_trend(customer)

    # Net score impact (positive = good, negative = bad)
    # Perfect history = +40, terrible history = -80
    if on_time_rate >= 0.99 and severity == 0:
        score_impact = +40.0
    elif on_time_rate >= 0.95 and severity <= 5:
        score_impact = +20.0
    elif on_time_rate >= 0.90 and severity <= 15:
        score_impact = 0.0
    elif on_time_rate >= 0.80 and severity <= 30:
        score_impact = -15.0
    elif on_time_rate >= 0.70 and severity <= 50:
        score_impact = -30.0
    else:
        score_impact = min(-80.0, -severity * 0.8)

    return {
        "customer_id": customer.id,
        "on_time_payments": customer.on_time_payments,
        "total_late_events": total_late,
        "late_30d_count": customer.late_payments_30d,
        "late_60d_count": customer.late_payments_60d,
        "late_90d_count": customer.late_payments_90d,
        "collections_count": customer.collections,
        "bankruptcies_count": customer.bankruptcies,
        "foreclosures_count": customer.foreclosures,
        "charge_offs_count": customer.charge_offs,
        "on_time_rate": round(on_time_rate, 4),
        "delinquency_score": round(delinquency_score, 2),
        "severity_score": round(severity, 2),
        "trend": trend,
        "trend_description": trend_description,
        "score_impact": round(score_impact, 2),
    }
