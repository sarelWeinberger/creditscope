#!/usr/bin/env python3
"""Seed the CreditScope SQLite database with synthetic customers."""

from backend.db.seed import seed_database


if __name__ == "__main__":
    seed_database()
