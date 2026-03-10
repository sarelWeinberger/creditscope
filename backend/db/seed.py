"""
Synthetic data generator for CreditScope customer database.

Generates 50+ realistic customers with diverse credit profiles,
realistic correlations, and edge cases.
"""

import random
from datetime import date, datetime, timedelta

from faker import Faker

from backend.db.models import Customer, LoanApplication, SessionLocal, init_db

fake = Faker()
Faker.seed(42)
random.seed(42)

# Credit profile templates
PROFILES = {
    "excellent": {
        "fico_range": (770, 850),
        "late_12m": (0, 0),
        "late_24m": (0, 1),
        "defaults": 0,
        "bankruptcies": 0,
        "collections": 0,
        "utilization_range": (0.02, 0.15),
        "history_years": (10, 30),
        "inquiries_6m": (0, 1),
    },
    "good": {
        "fico_range": (700, 769),
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
        "inquiries_6m": (1, 3),
    },
    "poor": {
        "fico_range": (550, 649),
        "late_12m": (2, 5),
        "late_24m": (4, 10),
        "defaults": (0, 2),
        "bankruptcies": 0,
        "collections": (0, 3),
        "utilization_range": (0.50, 0.85),
        "history_years": (2, 10),
        "inquiries_6m": (2, 5),
    },
    "very_poor": {
        "fico_range": (300, 549),
        "late_12m": (4, 8),
        "late_24m": (8, 15),
        "defaults": (1, 4),
        "bankruptcies": (0, 2),
        "collections": (1, 5),
        "utilization_range": (0.75, 1.0),
        "history_years": (1, 8),
        "inquiries_6m": (3, 8),
    },
}

# Distribution: more good/fair than extreme
PROFILE_WEIGHTS = {
    "excellent": 8,
    "good": 15,
    "fair": 14,
    "poor": 10,
    "very_poor": 5,
}

EMPLOYMENT_STATUSES = ["employed", "self_employed", "unemployed", "retired"]
EMPLOYMENT_WEIGHTS = [0.55, 0.20, 0.10, 0.15]
RESIDENTIAL_STATUSES = ["own", "rent", "mortgage"]


def _rand_range(val):
    if isinstance(val, tuple):
        return random.randint(val[0], val[1])
    return val


def _generate_customer(profile_name: str) -> dict:
    """Generate a single customer matching the given credit profile."""
    profile = PROFILES[profile_name]

    employment = random.choices(EMPLOYMENT_STATUSES, weights=EMPLOYMENT_WEIGHTS, k=1)[0]

    # Income varies by employment
    income_ranges = {
        "employed": (35000, 250000),
        "self_employed": (25000, 400000),
        "unemployed": (0, 20000),
        "retired": (20000, 150000),
    }
    income_min, income_max = income_ranges[employment]
    annual_income = round(random.uniform(income_min, income_max), 2)

    # Better credit profiles tend to have higher incomes (but not always)
    if profile_name in ("excellent", "good") and random.random() < 0.3:
        annual_income = round(random.uniform(80000, 350000), 2)

    # High income doesn't always mean good credit (edge case)
    monthly_income = annual_income / 12
    expense_ratio = random.uniform(0.3, 0.7)
    monthly_expenses = round(monthly_income * expense_ratio, 2)

    credit_history_years = _rand_range(profile["history_years"])
    total_credit_limit = round(random.uniform(5000, 100000), 2)
    util_min, util_max = profile["utilization_range"]
    utilization = random.uniform(util_min, util_max)
    total_credit_used = round(total_credit_limit * utilization, 2)

    has_mortgage = random.random() < 0.4
    has_auto = random.random() < 0.35
    has_student = random.random() < 0.25

    mortgage_balance = round(random.uniform(50000, 500000), 2) if has_mortgage else 0
    auto_balance = round(random.uniform(5000, 45000), 2) if has_auto else 0
    student_balance = round(random.uniform(10000, 120000), 2) if has_student else 0

    total_revolving = round(total_credit_used + random.uniform(0, 15000), 2)
    monthly_debt = round(
        (mortgage_balance * 0.005 + auto_balance * 0.02 + student_balance * 0.01
         + total_revolving * 0.02),
        2,
    )

    dob = fake.date_of_birth(minimum_age=22, maximum_age=75)

    # Risk notes for edge cases
    risk_notes = None
    if profile_name == "very_poor" and random.random() < 0.3:
        risk_notes = random.choice([
            "Recent bankruptcy filing — Chapter 7 discharged 6 months ago",
            "Identity fraud flag — disputed accounts under investigation",
            "Multiple collections accounts transferred to third-party agencies",
        ])
    elif profile_name == "fair" and random.random() < 0.2:
        risk_notes = random.choice([
            "Thin credit file — limited history, recently immigrated",
            "Recovering from medical debt — payment plan in place",
            "Recently divorced — joint accounts being separated",
        ])
    elif profile_name == "excellent" and random.random() < 0.15:
        risk_notes = "High net worth individual — assets significantly exceed reported income"

    return {
        "full_name": fake.name(),
        "ssn_last4": fake.numerify("####"),
        "date_of_birth": dob,
        "employment_status": employment,
        "employer_name": fake.company() if employment in ("employed", "self_employed") else None,
        "annual_income": annual_income,
        "monthly_expenses": monthly_expenses,
        "years_at_current_job": random.randint(0, min(20, credit_history_years + 5))
        if employment != "unemployed"
        else 0,
        "residential_status": random.choice(RESIDENTIAL_STATUSES),
        "years_at_address": random.randint(0, 25),
        "credit_history_years": credit_history_years,
        "num_open_accounts": random.randint(1, 15),
        "num_credit_cards": random.randint(0, 8),
        "total_credit_limit": total_credit_limit,
        "total_credit_used": total_credit_used,
        "num_late_payments_12m": _rand_range(profile["late_12m"]),
        "num_late_payments_24m": _rand_range(profile["late_24m"]),
        "num_defaults": _rand_range(profile["defaults"]),
        "num_bankruptcies": _rand_range(profile["bankruptcies"]),
        "num_collections": _rand_range(profile["collections"]),
        "has_mortgage": has_mortgage,
        "mortgage_balance": mortgage_balance,
        "has_auto_loan": has_auto,
        "auto_loan_balance": auto_balance,
        "has_student_loan": has_student,
        "student_loan_balance": student_balance,
        "total_revolving_debt": total_revolving,
        "monthly_debt_payments": monthly_debt,
        "num_hard_inquiries_6m": _rand_range(profile["inquiries_6m"]),
        "num_hard_inquiries_12m": _rand_range(profile["inquiries_6m"]) + random.randint(0, 3),
        "fico_score": random.randint(*profile["fico_range"]),
        "risk_notes": risk_notes,
    }


LOAN_TYPES = ["personal", "mortgage", "auto", "business", "student"]
LOAN_PURPOSES = {
    "personal": ["Debt consolidation", "Home improvement", "Medical expenses", "Vacation", "Wedding"],
    "mortgage": ["Primary residence purchase", "Investment property", "Refinance existing mortgage"],
    "auto": ["New vehicle purchase", "Used vehicle purchase", "Vehicle refinance"],
    "business": ["Working capital", "Equipment purchase", "Business expansion", "Inventory financing"],
    "student": ["Undergraduate tuition", "Graduate school", "Student loan refinance"],
}
COLLATERAL_MAP = {
    "personal": ("none", 0),
    "mortgage": ("real_estate", None),
    "auto": ("vehicle", None),
    "business": ("equipment", None),
    "student": ("none", 0),
}
STATUSES = ["pending", "approved", "rejected", "under_review"]


def _generate_loan(customer_id: int, fico: int) -> dict:
    """Generate a loan application correlated with customer's credit profile."""
    loan_type = random.choice(LOAN_TYPES)
    purpose = random.choice(LOAN_PURPOSES[loan_type])

    amount_ranges = {
        "personal": (2000, 50000),
        "mortgage": (100000, 800000),
        "auto": (10000, 75000),
        "business": (25000, 500000),
        "student": (5000, 100000),
    }
    min_amt, max_amt = amount_ranges[loan_type]
    amount = round(random.uniform(min_amt, max_amt), 2)

    term_ranges = {
        "personal": (12, 60),
        "mortgage": (120, 360),
        "auto": (24, 72),
        "business": (12, 120),
        "student": (60, 240),
    }
    term = random.choice(range(term_ranges[loan_type][0], term_ranges[loan_type][1] + 1, 12))

    # Interest rate inversely correlated with FICO
    base_rate = max(3.0, 18.0 - (fico - 300) * 0.025)
    rate = round(base_rate + random.uniform(-1, 2), 2)

    collateral_type, collateral_val = COLLATERAL_MAP[loan_type]
    if collateral_val is None:
        collateral_val = round(amount * random.uniform(0.7, 1.3), 2)

    # Status correlated with FICO
    if fico >= 750:
        status = random.choices(STATUSES, weights=[0.1, 0.7, 0.05, 0.15], k=1)[0]
    elif fico >= 650:
        status = random.choices(STATUSES, weights=[0.2, 0.4, 0.15, 0.25], k=1)[0]
    else:
        status = random.choices(STATUSES, weights=[0.15, 0.1, 0.5, 0.25], k=1)[0]

    return {
        "customer_id": customer_id,
        "loan_type": loan_type,
        "requested_amount": amount,
        "proposed_term_months": term,
        "proposed_interest_rate": rate,
        "collateral_type": collateral_type,
        "collateral_value": collateral_val if collateral_type != "none" else None,
        "purpose": purpose,
        "status": status,
    }


def seed_database():
    """Seed the database with synthetic customers and loan applications."""
    init_db()
    session = SessionLocal()

    try:
        # Check if already seeded
        if session.query(Customer).count() > 0:
            print("Database already seeded. Skipping.")
            return

        # Generate customers
        customers = []
        profile_pool = []
        for name, weight in PROFILE_WEIGHTS.items():
            profile_pool.extend([name] * weight)

        for i in range(52):
            profile = profile_pool[i % len(profile_pool)]
            data = _generate_customer(profile)
            customer = Customer(**data)
            session.add(customer)
            customers.append((customer, data["fico_score"]))

        session.flush()  # Get IDs assigned

        # Generate loan applications (10-15 active)
        loan_count = 0
        for customer, fico in customers:
            if random.random() < 0.28 and loan_count < 15:
                loan_data = _generate_loan(customer.id, fico)
                session.add(LoanApplication(**loan_data))
                loan_count += 1
                # Some customers have multiple loans
                if random.random() < 0.2 and loan_count < 15:
                    loan_data2 = _generate_loan(customer.id, fico)
                    session.add(LoanApplication(**loan_data2))
                    loan_count += 1

        session.commit()
        print(f"Seeded {len(customers)} customers and {loan_count} loan applications.")

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


if __name__ == "__main__":
    seed_database()
