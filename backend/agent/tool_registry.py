"""
Tool registration and dispatch for the CreditScope agent.
"""

from __future__ import annotations

import structlog

from backend.db.models import get_session
from backend.db.queries import (
    search_customers, get_customer_by_id, get_customer_loans,
    get_loan_by_id, add_customer_document,
)

logger = structlog.get_logger(__name__)


def _customer_to_dict(c) -> dict:
    """Convert a Customer ORM object to a dict."""
    return {
        "id": c.id,
        "full_name": c.full_name,
        "ssn_last4": c.ssn_last4,
        "date_of_birth": str(c.date_of_birth),
        "employment_status": c.employment_status,
        "employer_name": c.employer_name,
        "annual_income": c.annual_income,
        "monthly_expenses": c.monthly_expenses,
        "years_at_current_job": c.years_at_current_job,
        "residential_status": c.residential_status,
        "years_at_address": c.years_at_address,
        "credit_history_years": c.credit_history_years,
        "num_open_accounts": c.num_open_accounts,
        "num_credit_cards": c.num_credit_cards,
        "total_credit_limit": c.total_credit_limit,
        "total_credit_used": c.total_credit_used,
        "num_late_payments_12m": c.num_late_payments_12m,
        "num_late_payments_24m": c.num_late_payments_24m,
        "num_defaults": c.num_defaults,
        "num_bankruptcies": c.num_bankruptcies,
        "num_collections": c.num_collections,
        "has_mortgage": c.has_mortgage,
        "mortgage_balance": c.mortgage_balance,
        "has_auto_loan": c.has_auto_loan,
        "auto_loan_balance": c.auto_loan_balance,
        "has_student_loan": c.has_student_loan,
        "student_loan_balance": c.student_loan_balance,
        "total_revolving_debt": c.total_revolving_debt,
        "monthly_debt_payments": c.monthly_debt_payments,
        "num_hard_inquiries_6m": c.num_hard_inquiries_6m,
        "num_hard_inquiries_12m": c.num_hard_inquiries_12m,
        "fico_score": c.fico_score,
        "risk_notes": c.risk_notes,
    }


class ToolRegistry:
    """Registers and dispatches tool calls for the agent."""

    def __init__(self):
        self._tools: dict[str, callable] = {
            "lookup_customer": self._lookup_customer,
            "calculate_base_credit_score": self._calculate_base_credit_score,
            "calculate_loan_adjusted_score": self._calculate_loan_adjusted_score,
            "calculate_dti_ratio": self._calculate_dti_ratio,
            "evaluate_collateral": self._evaluate_collateral,
            "analyze_payment_history": self._analyze_payment_history,
            "compute_risk_weighted_score": self._compute_risk_weighted_score,
            "ingest_document_data": self._ingest_document_data,
        }

    async def execute(self, tool_name: str, args: dict) -> dict:
        """Execute a tool by name with given arguments."""
        handler = self._tools.get(tool_name)
        if not handler:
            logger.warning("unknown_tool", tool=tool_name)
            return {"error": f"Unknown tool: {tool_name}"}

        try:
            logger.info("executing_tool", tool=tool_name, args=args)
            result = await handler(**args)
            return result
        except Exception as e:
            logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return {"error": f"Tool execution failed: {str(e)}"}

    async def _lookup_customer(self, query: str, search_type: str = "fuzzy") -> dict:
        session = get_session()
        try:
            customers = search_customers(session, query, search_type)
            if not customers:
                return {"found": False, "message": f"No customers found matching '{query}'"}
            return {
                "found": True,
                "count": len(customers),
                "customers": [_customer_to_dict(c) for c in customers],
            }
        finally:
            session.close()

    async def _calculate_base_credit_score(self, customer_id: int) -> dict:
        from backend.tools.credit_score import calculate_base_credit_score
        session = get_session()
        try:
            customer = get_customer_by_id(session, customer_id)
            if not customer:
                return {"error": f"Customer {customer_id} not found"}
            return calculate_base_credit_score(customer)
        finally:
            session.close()

    async def _calculate_loan_adjusted_score(
        self,
        customer_id: int,
        loan_amount: float,
        loan_term_months: int,
        interest_rate: float,
        loan_type: str,
        collateral_type: str = "none",
        collateral_value: float = 0,
    ) -> dict:
        from backend.tools.loan_structure import calculate_loan_adjusted_score
        session = get_session()
        try:
            customer = get_customer_by_id(session, customer_id)
            if not customer:
                return {"error": f"Customer {customer_id} not found"}
            return calculate_loan_adjusted_score(
                customer,
                loan_amount=loan_amount,
                loan_term_months=loan_term_months,
                interest_rate=interest_rate,
                loan_type=loan_type,
                collateral_type=collateral_type,
                collateral_value=collateral_value,
            )
        finally:
            session.close()

    async def _calculate_dti_ratio(
        self, customer_id: int, additional_monthly_payment: float = 0
    ) -> dict:
        from backend.tools.debt_to_income import calculate_dti_ratio
        session = get_session()
        try:
            customer = get_customer_by_id(session, customer_id)
            if not customer:
                return {"error": f"Customer {customer_id} not found"}
            return calculate_dti_ratio(customer, additional_monthly_payment)
        finally:
            session.close()

    async def _evaluate_collateral(
        self, collateral_type: str, collateral_value: float, loan_amount: float
    ) -> dict:
        from backend.tools.collateral_eval import evaluate_collateral
        return evaluate_collateral(collateral_type, collateral_value, loan_amount)

    async def _analyze_payment_history(self, customer_id: int) -> dict:
        from backend.tools.payment_history import analyze_payment_history
        session = get_session()
        try:
            customer = get_customer_by_id(session, customer_id)
            if not customer:
                return {"error": f"Customer {customer_id} not found"}
            return analyze_payment_history(customer)
        finally:
            session.close()

    async def _compute_risk_weighted_score(
        self, customer_id: int, loan_application_id: int | None = None
    ) -> dict:
        from backend.tools.risk_adjustment import compute_risk_weighted_score
        session = get_session()
        try:
            customer = get_customer_by_id(session, customer_id)
            if not customer:
                return {"error": f"Customer {customer_id} not found"}
            loan = None
            if loan_application_id:
                loan = get_loan_by_id(session, loan_application_id)
            return compute_risk_weighted_score(customer, loan)
        finally:
            session.close()

    async def _ingest_document_data(
        self, customer_id: int, document_type: str, extracted_fields: dict
    ) -> dict:
        session = get_session()
        try:
            customer = get_customer_by_id(session, customer_id)
            if not customer:
                return {"error": f"Customer {customer_id} not found"}
            doc = add_customer_document(
                session, customer_id, document_type, "uploaded", extracted_fields
            )
            return {
                "success": True,
                "document_id": doc.id,
                "customer_id": customer_id,
                "document_type": document_type,
                "extracted_fields": extracted_fields,
            }
        finally:
            session.close()
