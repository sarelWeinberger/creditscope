"""
Synthetic customer data generator for CreditScope.

Generates 55+ realistic customers with diverse credit profiles,
realistic correlations, and edge cases.
"""

import random
from datetime import date, datetime, timedelta

from faker import Faker

from backend.db.models import (
    Base, Customer, LoanApplication, CustomerDocument,
    engine, init_db, get_session,
)

fake = Faker()
Faker.seed(42)
random.seed(42)


# Credit profile templates
PROFILES = {
    "excellent": {
        "fico_range": (780, 850),
        "late_12m": (0, 0),
        "late_24m": (0, 1),
        "defaults": 0,
        "bankruptcies": 0,
        "collections": 0,
        "utilization_range": (0.01, 0.15),
        "history_years": (10, 30),
        "inquiries_6m": (0, 1),
    },
    "good": {
        "fico_range": (700, 779),
        "late_12m": (0, 1),
        "late_24m": (0, 3),
        "defaults": 0,
        "bankruptcies": 0,
        "collections": 0,
        "utilization_range": (0.10, 0.30),
        "history_years": (5, 20),
        "inquiries_6m": (0, 2),
    },
    "fair": {
        "fico_range": (650, 699),
        "late_12m": (1, 3),
        "late_24m": (2, 6),
        "defaults": 0,
        "bankruptcies": 0,
        "collections": (0, 1),
        "utilization_range": (0.30, 0.60),
        "history_years": (3, 15),
        "inquiries_6m": (1, 4),
    },
    "poor": {
        "fico_range": (550, 649),
        "late_12m": (2, 6),
        "late_24m": (4, 10),
        "defaults": (0, 2),
        "bankruptcies": 0,
        "collections": (1, 3),
        "utilization_range": (0.50, 0.85),
        "history_years": (2, 10),
        "inquiries_6m": (2, 6),
    },
    "very_poor": {
        "fico_range": (300, 549),
        "late_12m": (4, 12),
        "late_24m": (8, 20),
        "defaults": (1, 5),
        "bankruptcies": (0, 2),
        "collections": (2, 6),
        "utilization_range": (0.70, 1.0),
        "history_years": (1, 8),
        "inquiries_6m": (3, 8),
    },
}

EMPLOYMENT_STATUSES = ["employed", "self_employed", "unemployed", "retired"]
RESIDENTIAL_STATUSES = ["own", "rent", "mortgage"]
LOAN_TYPES = ["personal", "mortgage", "auto", "business", "student"]
LOAN_STATUSES = ["pending", "approved", "rejected", "under_review"]
COLLATERAL_TYPES = ["real_estate", "vehicle", "equipment", "none"]


def _rand_range(val):
    """Return a random int from a tuple range or the value itself."""
    if isinstance(val, tuple):
        return random.randint(val[0], val[1])
    return val


def _rand_float_range(val):
    if isinstance(val, tuple):
        return round(random.uniform(val[0], val[1]), 4)
    return val


def generate_customer(profile_name: str) -> dict:
    """Generate a single customer record matching a credit profile."""
    profile = PROFILES[profile_name]
    fico = random.randint(*profile["fico_range"])

    # Income correlates loosely with credit quality but not perfectly
    base_income = {
        "excellent": random.randint(60000, 350000),
        "good": random.randint(45000, 200000),
        "fair": random.randint(30000, 120000),
        "poor": random.randint(20000, 80000),
        "very_poor": random.randint(15000, 60000),
    }[profile_name]

    # Some high earners have bad credit (lifestyle inflation)
    if random.random() < 0.15 and profile_name in ("fair", "poor"):
        base_income = random.randint(100000, 250000)

    employment = random.choice(EMPLOYMENT_STATUSES)
    if profile_name == "very_poor" and random.random() < 0.3:
        employment = "unemployed"
    if base_income > 200000 and random.random() < 0.3:
        employment = "self_employed"

    employer = fake.company() if employment in ("employed", "self_employed") else None
    years_job = random.randint(0, 25) if employment != "unemployed" else 0

    credit_limit = random.randint(5000, 150000)
    utilization = _rand_float_range(profile["utilization_range"])
    credit_used = round(credit_limit * utilization, 2)

    monthly_income = base_income / 12
    monthly_expenses = round(monthly_income * random.uniform(0.4, 0.85), 2)

    has_mortgage = random.random() < (0.6 if profile_name in ("excellent", "good") else 0.3)
    mortgage_bal = round(random.uniform(50000, 600000), 2) if has_mortgage else 0
    has_auto = random.random() < 0.5
    auto_bal = round(random.uniform(5000, 55000), 2) if has_auto else 0
    has_student = random.random() < 0.35
    student_bal = round(random.uniform(10000, 150000), 2) if has_student else 0

    revolving = round(credit_used + random.uniform(0, 20000), 2)
    monthly_debt = round(
        (mortgage_bal / 360 + auto_bal / 60 + student_bal / 120 + revolving * 0.02), 2
    )

    history_years = _rand_range(profile["history_years"])
    dob_year = 2026 - random.randint(max(22, history_years + 18), 75)

    num_late_24m = _rand_range(profile["late_24m"])
    num_late_12m = min(_rand_range(profile["late_12m"]), num_late_24m)

    risk_notes = None
    if profile_name == "very_poor" and random.random() < 0.3:
        risk_notes = random.choice([
            "Recent Chapter 7 bankruptcy filing",
            "Multiple accounts in collections — potential identity fraud flag",
            "Thin credit file — recent immigrant",
            "Chronic delinquency pattern — last 24 months",
        ])
    elif profile_name == "poor" and random.random() < 0.2:
        risk_notes = random.choice([
            "Recently discharged from bankruptcy — rebuilding credit",
            "High debt-to-income despite reasonable score",
            "Employment gap noted — returned to workforce 6 months ago",
        ])

    inq_6m = _rand_range(profile["inquiries_6m"])

    return {
        "full_name": fake.name(),
        "ssn_last4": fake.numerify("####"),
        "date_of_birth": date(dob_year, random.randint(1, 12), random.randint(1, 28)),
        "employment_status": employment,
        "employer_name": employer,
        "annual_income": float(base_income),
        "monthly_expenses": monthly_expenses,
        "years_at_current_job": years_job,
        "residential_status": random.choice(RESIDENTIAL_STATUSES),
        "years_at_address": random.randint(0, 20),
        "credit_history_years": history_years,
        "num_open_accounts": random.randint(2, 15),
        "num_credit_cards": random.randint(1, 10),
        "total_credit_limit": float(credit_limit),
        "total_credit_used": credit_used,
        "num_late_payments_12m": num_late_12m,
        "num_late_payments_24m": num_late_24m,
        "num_defaults": _rand_range(profile["defaults"]),
        "num_bankruptcies": _rand_range(profile["bankruptcies"]),
        "num_collections": _rand_range(profile["collections"]),
        "has_mortgage": has_mortgage,
        "mortgage_balance": mortgage_bal,
        "has_auto_loan": has_auto,
        "auto_loan_balance": auto_bal,
        "has_student_loan": has_student,
        "student_loan_balance": student_bal,
        "total_revolving_debt": revolving,
        "monthly_debt_payments": monthly_debt,
        "num_hard_inquiries_6m": inq_6m,
        "num_hard_inquiries_12m": inq_6m + random.randint(0, 3),
        "fico_score": fico,
        "risk_notes": risk_notes,
    }


def generate_loan_application(customer_id: int) -> dict:
    """Generate a loan application for a given customer."""
    loan_type = random.choice(LOAN_TYPES)

    amount_ranges = {
        "personal": (5000, 50000),
        "mortgage": (100000, 800000),
        "auto": (10000, 75000),
        "business": (25000, 500000),
        "student": (10000, 100000),
    }
    amount = round(random.uniform(*amount_ranges[loan_type]), 2)

    term_ranges = {
        "personal": (12, 60),
        "mortgage": (180, 360),
        "auto": (24, 84),
        "business": (12, 120),
        "student": (60, 240),
    }
    term = random.choice(range(term_ranges[loan_type][0], term_ranges[loan_type][1] + 1, 12))

    rate = round(random.uniform(3.5, 24.0), 2)

    collateral = "none"
    collateral_value = None
    if loan_type == "mortgage":
        collateral = "real_estate"
        collateral_value = round(amount * random.uniform(1.0, 1.5), 2)
    elif loan_type == "auto":
        collateral = "vehicle"
        collateral_value = round(amount * random.uniform(0.8, 1.3), 2)
    elif loan_type == "business" and random.random() < 0.5:
        collateral = random.choice(["real_estate", "equipment"])
        collateral_value = round(amount * random.uniform(0.6, 1.4), 2)

    purposes = {
        "personal": ["debt consolidation", "home improvement", "medical expenses", "vacation", "wedding"],
        "mortgage": ["primary residence purchase", "investment property", "refinance"],
        "auto": ["new vehicle purchase", "used vehicle purchase", "vehicle refinance"],
        "business": ["working capital", "equipment purchase", "expansion", "inventory"],
        "student": ["undergraduate tuition", "graduate program", "professional certification"],
    }

    return {
        "customer_id": customer_id,
        "loan_type": loan_type,
        "requested_amount": amount,
        "proposed_term_months": term,
        "proposed_interest_rate": rate,
        "collateral_type": collateral,
        "collateral_value": collateral_value,
        "purpose": random.choice(purposes[loan_type]),
        "status": random.choice(LOAN_STATUSES),
    }


def seed_database():
    """Seed the database with synthetic customer data."""
    init_db()
    session = get_session()

    # Clear existing data
    session.query(CustomerDocument).delete()
    session.query(LoanApplication).delete()
    session.query(Customer).delete()
    session.commit()

    # Distribution: 12 excellent, 12 good, 12 fair, 10 poor, 9 very_poor = 55
    distribution = {
        "excellent": 12,
        "good": 12,
        "fair": 12,
        "poor": 10,
        "very_poor": 9,
    }

    customer_ids = []
    for profile_name, count in distribution.items():
        for _ in range(count):
            data = generate_customer(profile_name)
            customer = Customer(**data)
            session.add(customer)
            session.flush()
            customer_ids.append(customer.id)

    session.commit()

    # Generate 12-15 active loan applications
    num_loans = random.randint(12, 15)
    loan_customer_ids = random.sample(customer_ids, min(num_loans, len(customer_ids)))

    for cid in loan_customer_ids:
        loan_data = generate_loan_application(cid)
        loan = LoanApplication(**loan_data)
        session.add(loan)

    session.commit()

    total_customers = session.query(Customer).count()
    total_loans = session.query(LoanApplication).count()
    session.close()

    print(f"Seeded {total_customers} customers and {total_loans} loan applications.")
    return total_customers, total_loans


if __name__ == "__main__":
    seed_database()
