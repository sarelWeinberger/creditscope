"""
Base credit score calculator using FICO-like methodology.

Weights:
- Payment History (35%): Late payments, defaults, bankruptcies, collections
- Amounts Owed (30%): Credit utilization ratio, total debt load
- Length of History (15%): Years of credit history
- New Credit (10%): Hard inquiries in last 6/12 months
- Credit Mix (10%): Variety of account types

Score range: 300-850
"""


def calculate_base_credit_score(customer) -> dict:
    """Calculate base credit score for a customer using FICO-like methodology."""
    # Payment History (35%) — score 0-100
    payment_score = _score_payment_history(customer)

    # Amounts Owed (30%) — score 0-100
    amounts_score = _score_amounts_owed(customer)

    # Length of History (15%) — score 0-100
    history_score = _score_history_length(customer)

    # New Credit (10%) — score 0-100
    new_credit_score = _score_new_credit(customer)

    # Credit Mix (10%) — score 0-100
    mix_score = _score_credit_mix(customer)

    # Weighted composite (0-100 scale)
    composite = (
        payment_score * 0.35
        + amounts_score * 0.30
        + history_score * 0.15
        + new_credit_score * 0.10
        + mix_score * 0.10
    )

    # Map to 300-850 range
    score = int(300 + (composite / 100) * 550)
    score = max(300, min(850, score))

    grade = _score_to_grade(score)
    factors = _identify_factors(customer, payment_score, amounts_score, history_score, new_credit_score, mix_score)

    return {
        "score": score,
        "grade": grade,
        "factors": factors,
        "breakdown": {
            "payment_history": round(payment_score, 1),
            "amounts_owed": round(amounts_score, 1),
            "length_of_history": round(history_score, 1),
            "new_credit": round(new_credit_score, 1),
            "credit_mix": round(mix_score, 1),
        },
    }


def _score_payment_history(customer) -> float:
    """Score payment history (0-100). Higher = better."""
    score = 100.0

    # Late payments in last 12 months (most impactful)
    late_12m = customer.num_late_payments_12m
    if late_12m >= 6:
        score -= 50
    elif late_12m >= 3:
        score -= 35
    elif late_12m >= 1:
        score -= 15

    # Late payments in 12-24 month window (less impactful)
    late_older = customer.num_late_payments_24m - customer.num_late_payments_12m
    if late_older >= 4:
        score -= 15
    elif late_older >= 2:
        score -= 8

    # Defaults
    if customer.num_defaults >= 3:
        score -= 30
    elif customer.num_defaults >= 1:
        score -= 20

    # Bankruptcies (severe)
    if customer.num_bankruptcies >= 2:
        score -= 40
    elif customer.num_bankruptcies >= 1:
        score -= 25

    # Collections
    if customer.num_collections >= 3:
        score -= 20
    elif customer.num_collections >= 1:
        score -= 10

    return max(0, score)


def _score_amounts_owed(customer) -> float:
    """Score amounts owed / utilization (0-100)."""
    score = 100.0

    # Credit utilization ratio
    if customer.total_credit_limit > 0:
        utilization = customer.total_credit_used / customer.total_credit_limit
    else:
        utilization = 0

    if utilization > 0.90:
        score -= 50
    elif utilization > 0.75:
        score -= 35
    elif utilization > 0.50:
        score -= 20
    elif utilization > 0.30:
        score -= 10
    elif utilization > 0.10:
        score -= 0  # Ideal range
    elif utilization == 0:
        score -= 5  # No utilization can be slightly negative

    # Total debt burden relative to income
    if customer.annual_income > 0:
        total_debt = (
            customer.mortgage_balance
            + customer.auto_loan_balance
            + customer.student_loan_balance
            + customer.total_revolving_debt
        )
        debt_to_income = total_debt / customer.annual_income
        if debt_to_income > 5:
            score -= 25
        elif debt_to_income > 3:
            score -= 15
        elif debt_to_income > 1.5:
            score -= 5

    return max(0, score)


def _score_history_length(customer) -> float:
    """Score length of credit history (0-100)."""
    years = customer.credit_history_years
    if years >= 20:
        return 100
    if years >= 15:
        return 90
    if years >= 10:
        return 75
    if years >= 7:
        return 60
    if years >= 5:
        return 45
    if years >= 3:
        return 30
    if years >= 1:
        return 15
    return 5


def _score_new_credit(customer) -> float:
    """Score new credit inquiries (0-100). Fewer inquiries = better."""
    score = 100.0

    inq_6m = customer.num_hard_inquiries_6m
    if inq_6m >= 6:
        score -= 40
    elif inq_6m >= 4:
        score -= 25
    elif inq_6m >= 2:
        score -= 10
    elif inq_6m == 1:
        score -= 5

    # Additional older inquiries
    inq_older = customer.num_hard_inquiries_12m - customer.num_hard_inquiries_6m
    if inq_older >= 4:
        score -= 10
    elif inq_older >= 2:
        score -= 5

    return max(0, score)


def _score_credit_mix(customer) -> float:
    """Score credit mix variety (0-100)."""
    types = 0
    if customer.num_credit_cards > 0:
        types += 1
    if customer.has_mortgage:
        types += 1
    if customer.has_auto_loan:
        types += 1
    if customer.has_student_loan:
        types += 1
    if customer.num_open_accounts > customer.num_credit_cards:
        types += 1  # Has non-card accounts (installment loans, etc.)

    if types >= 4:
        return 100
    if types == 3:
        return 80
    if types == 2:
        return 60
    if types == 1:
        return 35
    return 10


def _score_to_grade(score: int) -> str:
    """Convert numeric score to letter grade."""
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


def _identify_factors(customer, payment, amounts, history, new_credit, mix) -> list[str]:
    """Identify key positive and negative factors."""
    factors = []

    # Negative factors (sorted by impact)
    if payment < 50:
        factors.append("Significant delinquency history")
    elif payment < 75:
        factors.append("Recent late payments impacting score")

    if amounts < 50:
        factors.append("Very high credit utilization")
    elif amounts < 75:
        factors.append("Credit utilization above optimal range")

    if history < 40:
        factors.append("Short credit history")

    if new_credit < 60:
        factors.append("Too many recent credit inquiries")

    if mix < 50:
        factors.append("Limited credit mix")

    if customer.num_bankruptcies > 0:
        factors.append("Bankruptcy on record")

    if customer.num_collections > 0:
        factors.append("Accounts in collections")

    # Positive factors
    if payment >= 90:
        factors.append("Excellent payment history")
    if amounts >= 85:
        factors.append("Low credit utilization")
    if history >= 80:
        factors.append("Long established credit history")
    if customer.num_late_payments_12m == 0:
        factors.append("No recent late payments")

    return factors
