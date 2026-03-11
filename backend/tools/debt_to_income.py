"""Debt-to-Income ratio computation tool."""


def calculate_dti_ratio(customer, additional_monthly_payment: float = 0) -> dict:
    """
    Calculate Debt-to-Income ratio.

    Args:
        customer: Customer ORM object
        additional_monthly_payment: Optional new loan monthly payment

    Returns:
        DTI analysis dict
    """
    gross_monthly_income = customer.annual_income / 12

    if gross_monthly_income <= 0:
        return {
            "current_dti": None,
            "projected_dti": None,
            "gross_monthly_income": 0,
            "total_monthly_debt": customer.monthly_debt_payments,
            "additional_payment": additional_monthly_payment,
            "assessment": "Cannot calculate DTI — no income reported.",
        }

    current_dti = customer.monthly_debt_payments / gross_monthly_income
    projected_dti = (
        (customer.monthly_debt_payments + additional_monthly_payment) / gross_monthly_income
        if additional_monthly_payment > 0
        else None
    )

    effective_dti = projected_dti if projected_dti is not None else current_dti

    if effective_dti < 0.28:
        assessment = "Excellent DTI ratio. Well within recommended limits."
    elif effective_dti < 0.36:
        assessment = "Good DTI ratio. Within standard lending guidelines."
    elif effective_dti < 0.43:
        assessment = "Elevated DTI ratio. Approaching conventional lending limits. May require compensating factors."
    elif effective_dti < 0.50:
        assessment = "High DTI ratio. Exceeds conventional limits. Significant risk of payment stress."
    else:
        assessment = "Very high DTI ratio. Severe risk of default. Applicant may not be able to sustain payments."

    return {
        "current_dti": round(current_dti, 4),
        "projected_dti": round(projected_dti, 4) if projected_dti is not None else None,
        "gross_monthly_income": round(gross_monthly_income, 2),
        "total_monthly_debt": round(customer.monthly_debt_payments, 2),
        "additional_payment": round(additional_monthly_payment, 2),
        "assessment": assessment,
        "debt_breakdown": {
            "mortgage": round(customer.mortgage_balance / 360, 2) if customer.has_mortgage else 0,
            "auto_loan": round(customer.auto_loan_balance / 60, 2) if customer.has_auto_loan else 0,
            "student_loan": round(customer.student_loan_balance / 120, 2) if customer.has_student_loan else 0,
            "revolving_minimum": round(customer.total_revolving_debt * 0.02, 2),
        },
    }
