"""
Tool registry for the CreditScope agent.
Maps tool names to handler functions and manages execution.
"""
import json
import time
from typing import Any, Callable, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from db.queries import (
    get_customer_by_id,
    get_customer_loans,
    search_customers,
    create_customer_document,
)
from tools.credit_score import calculate_base_credit_score
from tools.loan_structure import calculate_loan_adjusted_score
from tools.debt_to_income import calculate_dti_ratio
from tools.collateral_eval import evaluate_collateral
from tools.payment_history import analyze_payment_history
from tools.risk_adjustment import compute_risk_weighted_score


class ToolError(Exception):
    """Raised when a tool execution fails."""
    pass


class ToolRegistry:
    """
    Registry of all available tools with async execution support.
    Each tool is registered with a name and an async handler function.
    """

    def __init__(self, session_factory=None):
        self._handlers: Dict[str, Callable] = {}
        self._session_factory = session_factory
        self._register_all()

    def _register_all(self):
        """Register all tool handlers."""
        self._handlers["lookup_customer"] = self._lookup_customer
        self._handlers["calculate_base_credit_score"] = self._calculate_base_credit_score
        self._handlers["calculate_loan_adjusted_score"] = self._calculate_loan_adjusted_score
        self._handlers["calculate_dti_ratio"] = self._calculate_dti_ratio
        self._handlers["evaluate_collateral"] = self._evaluate_collateral
        self._handlers["analyze_payment_history"] = self._analyze_payment_history
        self._handlers["compute_risk_weighted_score"] = self._compute_risk_weighted_score
        self._handlers["ingest_document_data"] = self._ingest_document_data

    def _get_session(self) -> AsyncSession:
        """Create a new database session."""
        if self._session_factory is None:
            raise ToolError("Database session factory not configured.")
        return self._session_factory()

    async def execute_tool(
        self, tool_name: str, tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a named tool with the given input.

        Returns:
            dict with keys: result, success, error (if any), duration_ms
        """
        start = time.monotonic()

        handler = self._handlers.get(tool_name)
        if handler is None:
            return {
                "result": None,
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "duration_ms": 0.0,
            }

        try:
            result = await handler(**tool_input)
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            return {
                "result": result,
                "success": True,
                "error": None,
                "duration_ms": duration_ms,
            }
        except ToolError as e:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            return {
                "result": None,
                "success": False,
                "error": str(e),
                "duration_ms": duration_ms,
            }
        except Exception as e:
            duration_ms = round((time.monotonic() - start) * 1000, 2)
            return {
                "result": None,
                "success": False,
                "error": f"Tool execution error: {type(e).__name__}: {e}",
                "duration_ms": duration_ms,
            }

    def format_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Format a tool result as a JSON string for the model."""
        if not result["success"]:
            return json.dumps({"error": result["error"]})
        return json.dumps(result["result"], default=str, indent=2)

    # ─── Tool Handlers ────────────────────────────────────────────────────────

    async def _lookup_customer(
        self, query: str, search_type: str = "fuzzy"
    ) -> Dict[str, Any]:
        async with self._get_session() as session:
            if search_type == "id":
                try:
                    cid = int(query)
                    customer = await get_customer_by_id(session, cid)
                    if customer is None:
                        raise ToolError(f"No customer found with ID {cid}")
                    customers = [customer]
                except ValueError:
                    raise ToolError(f"Invalid customer ID: {query}")
            else:
                customers = await search_customers(session, query, search_type)

            if not customers:
                raise ToolError(f"No customers found matching '{query}'")

            result = []
            for c in customers[:5]:
                result.append(_customer_to_dict(c))

            if len(result) == 1:
                return result[0]
            return {"matches": result, "count": len(result)}

    async def _calculate_base_credit_score(self, customer_id: int) -> Dict[str, Any]:
        async with self._get_session() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer is None:
                raise ToolError(f"Customer {customer_id} not found")
            return calculate_base_credit_score(customer)

    async def _calculate_loan_adjusted_score(
        self,
        customer_id: int,
        loan_amount: float,
        loan_type: str,
        term_months: int,
        collateral_type: str = "none",
        collateral_value: float = 0.0,
        loan_purpose: str = "",
        additional_monthly_payment: Optional[float] = None,
    ) -> Dict[str, Any]:
        async with self._get_session() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer is None:
                raise ToolError(f"Customer {customer_id} not found")

            loan_params = {
                "loan_amount": loan_amount,
                "loan_type": loan_type,
                "term_months": term_months,
                "collateral_type": collateral_type,
                "collateral_value": collateral_value,
                "loan_purpose": loan_purpose,
                "additional_monthly_payment": additional_monthly_payment,
            }
            return calculate_loan_adjusted_score(customer, loan_params)

    async def _calculate_dti_ratio(
        self, customer_id: int, additional_monthly_payment: float = 0.0
    ) -> Dict[str, Any]:
        async with self._get_session() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer is None:
                raise ToolError(f"Customer {customer_id} not found")
            return calculate_dti_ratio(customer, additional_monthly_payment)

    async def _evaluate_collateral(
        self,
        collateral_type: str,
        collateral_value: float,
        loan_amount: float,
    ) -> Dict[str, Any]:
        return evaluate_collateral(collateral_type, collateral_value, loan_amount)

    async def _analyze_payment_history(self, customer_id: int) -> Dict[str, Any]:
        async with self._get_session() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer is None:
                raise ToolError(f"Customer {customer_id} not found")
            return analyze_payment_history(customer)

    async def _compute_risk_weighted_score(
        self, customer_id: int, loan_application_id: Optional[int] = None
    ) -> Dict[str, Any]:
        async with self._get_session() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer is None:
                raise ToolError(f"Customer {customer_id} not found")

            loan = None
            if loan_application_id is not None:
                from db.queries import get_loan_by_id
                loan = await get_loan_by_id(session, loan_application_id)

            return compute_risk_weighted_score(customer, loan)

    async def _ingest_document_data(
        self,
        customer_id: int,
        document_type: str,
        file_name: str,
        parsed_data: Optional[Dict[str, Any]] = None,
        update_customer: bool = False,
    ) -> Dict[str, Any]:
        async with self._get_session() as session:
            customer = await get_customer_by_id(session, customer_id)
            if customer is None:
                raise ToolError(f"Customer {customer_id} not found")

            doc_data = {
                "customer_id": customer_id,
                "document_type": document_type,
                "file_name": file_name,
                "is_parsed": parsed_data is not None,
                "parsed_data": json.dumps(parsed_data) if parsed_data else None,
            }

            doc = await create_customer_document(session, doc_data)
            await session.commit()

            result = {
                "document_id": doc.id,
                "customer_id": customer_id,
                "document_type": document_type,
                "file_name": file_name,
                "is_parsed": doc.is_parsed,
                "status": "ingested",
            }

            if parsed_data:
                result["extracted_fields"] = list(parsed_data.keys())

            return result


def _customer_to_dict(customer) -> Dict[str, Any]:
    """Serialize a Customer ORM object to a dict."""
    return {
        "id": customer.id,
        "full_name": f"{customer.first_name} {customer.last_name}",
        "first_name": customer.first_name,
        "last_name": customer.last_name,
        "email": customer.email,
        "phone": customer.phone,
        "date_of_birth": str(customer.date_of_birth) if customer.date_of_birth else None,
        "address": customer.address,
        "city": customer.city,
        "state": customer.state,
        "zip_code": customer.zip_code,
        "employment_status": customer.employment_status,
        "employer_name": customer.employer_name,
        "annual_income": customer.annual_income,
        "monthly_income": customer.monthly_income,
        "years_employed": customer.years_employed,
        "job_title": customer.job_title,
        "credit_score": customer.credit_score,
        "credit_history_years": customer.credit_history_years,
        "num_credit_accounts": customer.num_credit_accounts,
        "num_open_accounts": customer.num_open_accounts,
        "total_credit_limit": customer.total_credit_limit,
        "total_credit_used": customer.total_credit_used,
        "credit_utilization_ratio": customer.credit_utilization_ratio,
        "total_debt": customer.total_debt,
        "monthly_debt_payments": customer.monthly_debt_payments,
        "mortgage_balance": customer.mortgage_balance,
        "auto_loan_balance": customer.auto_loan_balance,
        "student_loan_balance": customer.student_loan_balance,
        "credit_card_balance": customer.credit_card_balance,
        "on_time_payments": customer.on_time_payments,
        "late_payments_30d": customer.late_payments_30d,
        "late_payments_60d": customer.late_payments_60d,
        "late_payments_90d": customer.late_payments_90d,
        "collections": customer.collections,
        "bankruptcies": customer.bankruptcies,
        "foreclosures": customer.foreclosures,
        "charge_offs": customer.charge_offs,
        "hard_inquiries_6m": customer.hard_inquiries_6m,
        "hard_inquiries_12m": customer.hard_inquiries_12m,
        "checking_balance": customer.checking_balance,
        "savings_balance": customer.savings_balance,
        "investment_balance": customer.investment_balance,
        "property_value": customer.property_value,
        "vehicle_value": customer.vehicle_value,
        "is_active": customer.is_active,
    }
