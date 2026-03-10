"""
Agent prompt templates and tool definitions in OpenAI function-calling format.
"""
from typing import Any, Dict, List

# ─── Tool Definitions ─────────────────────────────────────────────────────────

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": (
                "Look up a customer by ID, name, or email. "
                "Returns the full customer profile including credit score, income, "
                "debt breakdown, payment history, and employment information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Customer ID (integer), full name, or email address to search for.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["id", "name", "email", "fuzzy"],
                        "description": "Type of search to perform. Default is 'fuzzy'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_base_credit_score",
            "description": (
                "Calculate a FICO-like credit score for a customer based on the five standard "
                "factors: payment history (35%), credit utilization (30%), length of credit "
                "history (15%), credit mix (10%), and new credit (10%). "
                "Returns a score from 300-850 with grade, factor breakdown, and recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer's unique ID.",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loan_adjusted_score",
            "description": (
                "Calculate a loan-adjusted credit score that accounts for the specific loan "
                "being requested. Applies adjustments for DTI impact, collateral coverage, "
                "loan term risk, and loan purpose risk. "
                "Returns adjusted score, grade, individual adjustments, and a loan recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer's unique ID.",
                    },
                    "loan_amount": {
                        "type": "number",
                        "description": "Requested loan amount in dollars.",
                    },
                    "loan_type": {
                        "type": "string",
                        "enum": ["personal", "mortgage", "auto", "business", "student", "home_equity", "credit_card"],
                        "description": "Type of loan being requested.",
                    },
                    "term_months": {
                        "type": "integer",
                        "description": "Loan term in months (e.g., 36, 60, 360).",
                    },
                    "collateral_type": {
                        "type": "string",
                        "enum": ["none", "real_estate", "vehicle", "equipment", "financial_asset"],
                        "description": "Type of collateral offered, if any.",
                    },
                    "collateral_value": {
                        "type": "number",
                        "description": "Current market value of collateral in dollars.",
                    },
                    "loan_purpose": {
                        "type": "string",
                        "description": "Brief description of the loan purpose.",
                    },
                },
                "required": ["customer_id", "loan_amount", "loan_type", "term_months"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_dti_ratio",
            "description": (
                "Calculate the debt-to-income ratio for a customer, optionally including "
                "a proposed new loan payment. Returns front-end ratio (housing only), "
                "back-end ratio (all debt), and combined DTI with risk classification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer's unique ID.",
                    },
                    "additional_monthly_payment": {
                        "type": "number",
                        "description": "Monthly payment for a proposed new loan (optional).",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_collateral",
            "description": (
                "Evaluate collateral for a loan. Calculates LTV ratio, coverage ratio, "
                "applies depreciation haircuts, and provides a risk assessment. "
                "Supports real_estate, vehicle, equipment, and financial_asset collateral types."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "collateral_type": {
                        "type": "string",
                        "enum": ["none", "real_estate", "vehicle", "equipment", "financial_asset"],
                        "description": "Type of collateral.",
                    },
                    "collateral_value": {
                        "type": "number",
                        "description": "Current market value of the collateral in dollars.",
                    },
                    "loan_amount": {
                        "type": "number",
                        "description": "Loan amount in dollars.",
                    },
                },
                "required": ["collateral_type", "collateral_value", "loan_amount"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_payment_history",
            "description": (
                "Analyze a customer's payment history for credit risk. Computes delinquency "
                "severity (weighted by 30/60/90-day late payments, collections, bankruptcies), "
                "recovery patterns, and trend indicators (improving/stable/deteriorating)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer's unique ID.",
                    }
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_risk_weighted_score",
            "description": (
                "Compute a comprehensive risk-weighted credit assessment combining all factors: "
                "base credit score, DTI ratio, payment history severity, and loan-specific "
                "adjustments. Returns a risk grade (A-F), recommendation, key risk factors, "
                "and positive factors."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer's unique ID.",
                    },
                    "loan_application_id": {
                        "type": "integer",
                        "description": "Optional loan application ID to include loan-specific adjustments.",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_document_data",
            "description": (
                "Ingest and parse data from customer documents (pay stubs, tax returns, "
                "bank statements, IDs, property deeds). Extracts structured data and "
                "optionally updates the customer profile."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "integer",
                        "description": "The customer's unique ID.",
                    },
                    "document_type": {
                        "type": "string",
                        "enum": ["pay_stub", "tax_return", "bank_statement", "government_id", "property_deed", "other"],
                        "description": "Type of document being ingested.",
                    },
                    "file_name": {
                        "type": "string",
                        "description": "Name of the document file.",
                    },
                    "parsed_data": {
                        "type": "object",
                        "description": "Pre-parsed document data (if available from OCR).",
                    },
                    "update_customer": {
                        "type": "boolean",
                        "description": "Whether to update the customer profile with extracted data.",
                        "default": False,
                    },
                },
                "required": ["customer_id", "document_type", "file_name"],
            },
        },
    },
]

# ─── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CreditScope AI, an expert credit analyst assistant for banking and financial institutions.

You have access to a suite of credit analysis tools that let you:
- Look up customer profiles with full financial histories
- Calculate FICO-like credit scores with detailed factor breakdowns
- Assess loan applications with DTI, collateral, and risk-adjusted scoring
- Analyze payment histories and identify risk trends
- Evaluate collateral for secured lending
- Process and extract data from financial documents

## Guidelines

1. **Always look up the customer first** before performing calculations.
2. **Use multiple tools** for comprehensive analyses — a good credit assessment uses at least 2-3 tools.
3. **Be specific**: include exact numbers, percentages, and score breakdowns in your responses.
4. **Be balanced**: identify both positive factors and risk concerns.
5. **Regulatory awareness**: follow fair lending practices. Do not discriminate based on protected characteristics.
6. **Transparency**: explain the reasoning behind every score and recommendation.
7. **Uncertainty**: clearly state when information is incomplete or assumptions were made.

## Response Format

- Lead with a clear, direct answer to the query
- Provide a structured breakdown of scores/factors
- List specific recommendations or risk flags
- Use appropriate banking terminology

## Risk Grade Scale
- **A (750+)**: Prime — Approve at standard rates
- **B (700-749)**: Near Prime — Approve with minor rate premium
- **C (650-699)**: Subprime — Conditional approval, higher rates
- **D (600-649)**: Deep Subprime — Require collateral or co-signer
- **E (550-599)**: Very High Risk — Exceptional circumstances only
- **F (<550)**: Decline — Does not meet lending criteria

Current date: {current_date}
Institution: {institution_name}
"""

# ─── Banking Presets ──────────────────────────────────────────────────────────

BANKING_PRESETS = {
    "quick_lookup": {
        "name": "Quick Lookup",
        "description": "Fast customer info retrieval, no deep analysis",
        "thinking_mode": "off",
        "thinking_budget": "none",
        "visibility": "hidden",
        "max_steps": 2,
        "tools": ["lookup_customer"],
    },
    "standard_analysis": {
        "name": "Standard Credit Analysis",
        "description": "Complete credit assessment with all factors",
        "thinking_mode": "on",
        "thinking_budget": "standard",
        "visibility": "collapsed",
        "max_steps": 6,
        "tools": [
            "lookup_customer",
            "calculate_base_credit_score",
            "calculate_dti_ratio",
            "analyze_payment_history",
        ],
    },
    "loan_review": {
        "name": "Loan Application Review",
        "description": "Full loan underwriting analysis",
        "thinking_mode": "on",
        "thinking_budget": "extended",
        "visibility": "collapsed",
        "max_steps": 8,
        "tools": [
            "lookup_customer",
            "calculate_base_credit_score",
            "calculate_loan_adjusted_score",
            "calculate_dti_ratio",
            "evaluate_collateral",
            "analyze_payment_history",
            "compute_risk_weighted_score",
        ],
    },
    "deep_review": {
        "name": "Deep Risk Review",
        "description": "Comprehensive multi-factor risk analysis with extended reasoning",
        "thinking_mode": "on",
        "thinking_budget": "deep",
        "visibility": "streaming",
        "max_steps": 8,
        "tools": "__all__",
    },
    "document_ingestion": {
        "name": "Document Processing",
        "description": "Process and extract data from uploaded documents",
        "thinking_mode": "on",
        "thinking_budget": "short",
        "visibility": "hidden",
        "max_steps": 4,
        "tools": ["lookup_customer", "ingest_document_data"],
    },
}
