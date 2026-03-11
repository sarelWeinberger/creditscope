"""Payment history analyzer tool."""


def analyze_payment_history(customer) -> dict:
    """
    Deep analysis of a customer's payment history pattern.

    Returns delinquency trends, severity scoring, and recovery patterns.
    """
    late_12m = customer.num_late_payments_12m
    late_24m = customer.num_late_payments_24m
    late_older = late_24m - late_12m
    defaults = customer.num_defaults
    bankruptcies = customer.num_bankruptcies
    collections = customer.num_collections

    # Severity score (0-100, higher = worse)
    severity = 0
    severity += min(late_12m * 8, 40)    # Recent late payments
    severity += min(late_older * 3, 15)   # Older late payments
    severity += min(defaults * 12, 30)    # Defaults
    severity += min(bankruptcies * 20, 30) # Bankruptcies
    severity += min(collections * 8, 20)  # Collections
    severity = min(severity, 100)

    # Delinquency trend
    if late_12m == 0 and late_older == 0:
        trend = "clean"
        trend_description = "No delinquencies found. Clean payment history."
    elif late_12m == 0 and late_older > 0:
        trend = "improving"
        trend_description = "Past delinquencies with recent improvement. No late payments in last 12 months."
    elif late_12m > 0 and late_12m > late_older:
        trend = "worsening"
        trend_description = "Delinquency pattern is worsening. More late payments recently than previously."
    elif late_12m > 0 and late_12m <= late_older:
        trend = "stable_delinquent"
        trend_description = "Ongoing delinquency pattern at similar rate."
    else:
        trend = "mixed"
        trend_description = "Mixed payment pattern."

    # Recovery pattern
    if bankruptcies > 0 and late_12m == 0:
        recovery = "post_bankruptcy_recovery"
        recovery_description = "Recovering from bankruptcy with clean recent payments."
    elif defaults > 0 and late_12m == 0:
        recovery = "post_default_recovery"
        recovery_description = "Previous defaults but currently maintaining payments."
    elif severity < 10:
        recovery = "no_recovery_needed"
        recovery_description = "Minimal delinquency history. No recovery pattern applicable."
    elif trend == "improving":
        recovery = "active_recovery"
        recovery_description = "Actively improving payment behavior."
    else:
        recovery = "no_recovery"
        recovery_description = "No clear recovery pattern observed."

    # Risk level
    if severity < 10:
        risk_level = "minimal"
    elif severity < 25:
        risk_level = "low"
    elif severity < 45:
        risk_level = "moderate"
    elif severity < 70:
        risk_level = "high"
    else:
        risk_level = "severe"

    return {
        "delinquency_trend": trend,
        "trend_description": trend_description,
        "severity_score": round(severity, 1),
        "recovery_pattern": recovery,
        "recovery_description": recovery_description,
        "late_payment_details": {
            "last_12_months": late_12m,
            "months_13_to_24": late_older,
            "total_24_months": late_24m,
            "defaults": defaults,
            "bankruptcies": bankruptcies,
            "collections": collections,
        },
        "risk_level": risk_level,
    }
