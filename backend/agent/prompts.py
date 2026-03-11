"""
System prompts and tool definitions for the CreditScope agent.
"""

SYSTEM_PROMPT = """You are CreditScope, an AI credit analysis assistant for bankers.
You have access to a customer database and credit scoring tools.

When a banker asks about a customer's credit rating, you should:
1. Look up the customer in the database
2. Retrieve their credit parameters
3. If a specific loan structure is mentioned, use the loan scoring tools
4. Calculate an adjusted credit score based on all available data
5. Provide a clear, structured assessment

Always explain your reasoning. If data is missing, say so.
Never fabricate financial data.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_customer",
            "description": "Look up a customer by name, ID, or partial match. Returns full customer profile with credit parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Customer name, ID, or search term"},
                    "search_type": {"type": "string", "enum": ["name", "id", "fuzzy"], "default": "fuzzy"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_base_credit_score",
            "description": "Calculate the base credit score for a customer using FICO-like methodology. Factors: payment history (35%), amounts owed (30%), length of history (15%), new credit (10%), credit mix (10%).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_loan_adjusted_score",
            "description": "Adjust credit score based on a proposed loan structure. Factors in: DTI impact, collateral coverage ratio, loan-to-value, term risk, and purpose risk weight.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "loan_amount": {"type": "number"},
                    "loan_term_months": {"type": "integer"},
                    "interest_rate": {"type": "number"},
                    "loan_type": {"type": "string", "enum": ["personal", "mortgage", "auto", "business", "student"]},
                    "collateral_type": {"type": "string", "enum": ["real_estate", "vehicle", "equipment", "none"], "default": "none"},
                    "collateral_value": {"type": "number", "default": 0}
                },
                "required": ["customer_id", "loan_amount", "loan_term_months", "interest_rate", "loan_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_dti_ratio",
            "description": "Calculate Debt-to-Income ratio for a customer, optionally including a proposed new loan payment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "additional_monthly_payment": {"type": "number", "default": 0}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_collateral",
            "description": "Evaluate collateral adequacy for a loan. Returns LTV ratio, coverage ratio, and risk assessment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "collateral_type": {"type": "string"},
                    "collateral_value": {"type": "number"},
                    "loan_amount": {"type": "number"}
                },
                "required": ["collateral_type", "collateral_value", "loan_amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_payment_history",
            "description": "Deep analysis of a customer's payment history pattern. Returns delinquency trends, severity scoring, and recovery patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_risk_weighted_score",
            "description": "Final composite risk score combining all factors: base credit score, DTI, collateral, payment history, and loan structure. Returns a risk grade (A through F) with detailed breakdown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "loan_application_id": {"type": "integer", "description": "Optional: specific loan application to evaluate"}
                },
                "required": ["customer_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ingest_document_data",
            "description": "Extract and store structured data from an uploaded document image (pay stub, tax return, bank statement, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "integer"},
                    "document_type": {"type": "string", "enum": ["pay_stub", "tax_return", "bank_statement", "id", "property_deed"]},
                    "extracted_fields": {"type": "object", "description": "Key-value pairs of extracted data"}
                },
                "required": ["customer_id", "document_type", "extracted_fields"]
            }
        }
    }
]
