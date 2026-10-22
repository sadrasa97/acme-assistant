"""Schema creation and deterministic, idempotent seed data for the sample business."""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS customers (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    country      TEXT NOT NULL,
    created_at   TEXT NOT NULL            -- ISO date (YYYY-MM-DD): account creation date
);
CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    amount       REAL NOT NULL CHECK (amount >= 0),   -- order value in USD
    status       TEXT NOT NULL CHECK (status IN ('completed','pending','cancelled','refunded')),
    order_date   TEXT NOT NULL            -- ISO date (YYYY-MM-DD)
);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
"""

_COMPANIES = [
    "Northwind", "Contoso", "Globex", "Initech", "Umbrella", "Hooli", "Stark", "Wayne", "Wonka", "Soylent",
    "Cyberdyne", "Tyrell", "Aperture", "Vandelay", "Dunder", "Pied Piper", "Massive", "Oscorp", "Gringotts", "Monarch",
    "Acme Retail", "Blue Sun", "Duff", "Los Pollos", "Prestige", "Sterling", "Bluth", "Kwik", "Nakatomi", "Gekko",
    "Rekall", "Virtucon", "Zorg", "Bubba", "Planet", "Spacely", "Cogswell", "Krusty", "Lexcorp", "Waystar",
]
_COUNTRIES = ["Germany", "United States", "United Kingdom", "France", "Canada", "Netherlands", "Spain", "Australia"]
_COUNTRY_WEIGHTS = [5, 8, 5, 4, 4, 3, 3, 2]
_STATUSES = ["completed", "pending", "cancelled", "refunded"]
_STATUS_WEIGHTS = [62, 14, 14, 10]

SEED = 20260101


def connect_rw(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def seed_database(path: Path) -> dict[str, int]:
    """Create tables and (re)load the sample data. Safe to run repeatedly: results are identical."""
    rng = random.Random(SEED)
    conn = connect_rw(path)
    try:
        with conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute("DELETE FROM orders")
            conn.execute("DELETE FROM customers")

            base = date(2024, 1, 1)
            customers = []
            for i, company in enumerate(_COMPANIES, start=1):
                country = rng.choices(_COUNTRIES, _COUNTRY_WEIGHTS)[0]
                created = base + timedelta(days=rng.randint(0, 540))
                customers.append((i, f"{company} GmbH" if country == "Germany" else f"{company} Inc", country, created.isoformat()))
            conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", customers)

            orders = []
            order_id = 1
            for cid, _, _, created_iso in customers:
                created = date.fromisoformat(created_iso)
                for _ in range(rng.randint(2, 9)):
                    order_date = created + timedelta(days=rng.randint(0, 400))
                    amount = round(rng.choice([199, 349, 499, 899, 1299, 2499, 4999, 9999]) * rng.uniform(0.8, 1.3), 2)
                    status = rng.choices(_STATUSES, _STATUS_WEIGHTS)[0]
                    orders.append((order_id, cid, amount, status, order_date.isoformat()))
                    order_id += 1
            conn.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
        return {
            "customers": conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0],
            "orders": conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
        }
    finally:
        conn.close()
