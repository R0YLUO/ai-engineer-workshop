"""The internal "database" for Swift Freight Co.

Everything lives in plain Python variables (in memory). The only thing that is
persisted to disk is the chat message list, so the chat app (a static web page)
can read it.

Invoice maths (how every invoice line SHOULD be calculated):

    line_total = distance_km * contracted_rate_per_km[service_level]
               + sum(surcharges that apply)
    invoice_total = sum(line_totals) * (1 - contracted_discount_pct / 100)

Some invoices below were calculated with the WRONG values on purpose - these are
the disputes the agent has to investigate.
"""

import json
from datetime import datetime
from pathlib import Path

CUSTOMERS = {
    "C001": {
        "customer_id": "C001",
        "name": "Acme Retail",
        "contact_email": "accounts@acmeretail.com.au",
        "contract": {
            "rate_per_km": {"standard": 2.50, "express": 4.00},
            "discount_pct": 0,
            "surcharges": {"refrigerated": 0, "heavy_item": 40.00},
        },
    },
    "C002": {
        "customer_id": "C002",
        "name": "Bluegum Pharmacy",
        "contact_email": "ops@bluegumpharmacy.com.au",
        "contract": {
            "rate_per_km": {"standard": 2.80, "express": 4.50},
            "discount_pct": 0,
            "surcharges": {"refrigerated": 25.00, "heavy_item": 40.00},
        },
    },
    "C003": {
        "customer_id": "C003",
        "name": "Coastal Furniture",
        "contact_email": "finance@coastalfurniture.com.au",
        "contract": {
            "rate_per_km": {"standard": 3.10, "express": 5.00},
            "discount_pct": 10,
            "surcharges": {"refrigerated": 0, "heavy_item": 40.00},
        },
    },
}

BOOKINGS = {
    "C001": [
        {
            "booking_id": "B-2001",
            "date": "2026-10-03",
            "route": "Sydney -> Newcastle",
            "distance_km": 120,
            "service_level_booked": "standard",
            "weight_kg": 45,
            "invoice": {
                "invoice_id": "INV-1042",
                "rate_per_km_applied": 4.00,
                "surcharges_applied": [],
                "discount_pct_applied": 0,
                "total": 480.00,
            },
        },
        {
            "booking_id": "B-2002",
            "date": "2026-10-09",
            "route": "Sydney -> Wollongong",
            "distance_km": 85,
            "service_level_booked": "express",
            "weight_kg": 30,
            "invoice": {
                "invoice_id": "INV-1043",
                "rate_per_km_applied": 4.00,
                "surcharges_applied": [],
                "discount_pct_applied": 0,
                "total": 340.00,
            },
        },
    ],
    "C002": [
        {
            "booking_id": "B-3001",
            "date": "2026-10-06",
            "route": "Melbourne -> Geelong",
            "distance_km": 75,
            "service_level_booked": "express",
            "weight_kg": 12,
            "refrigerated": True,
            "invoice": {
                "invoice_id": "INV-1047",
                "rate_per_km_applied": 4.50,
                "surcharges_applied": [{"type": "refrigerated", "amount": 25.00}],
                "discount_pct_applied": 0,
                "total": 362.50,
            },
        },
    ],
    "C003": [
        {
            "booking_id": "B-4001",
            "date": "2026-10-12",
            "route": "Brisbane -> Gold Coast",
            "distance_km": 80,
            "service_level_booked": "standard",
            "weight_kg": 180,
            "invoice": {
                "invoice_id": "INV-1051",
                "rate_per_km_applied": 3.10,
                "surcharges_applied": [{"type": "heavy_item", "amount": 40.00}],
                "discount_pct_applied": 0,
                "total": 288.00,
            },
        },
    ],
}

# The chat app reads this file. It is the only "table" stored on disk.
MESSAGES_FILE = Path(__file__).parent / "chat_app" / "messages.json"


def find_customers(name: str) -> list[dict]:
    """Case-insensitive partial match on customer name."""
    return [c for c in CUSTOMERS.values() if name.lower() in c["name"].lower()]


def get_bookings(customer_id: str) -> list[dict]:
    return BOOKINGS.get(customer_id, [])


def read_messages() -> list[dict]:
    if not MESSAGES_FILE.exists():
        return []
    return json.loads(MESSAGES_FILE.read_text())


def add_message(text: str, author: str = "Finance Agent") -> dict:
    messages = read_messages()
    message = {
        "id": len(messages) + 1,
        "author": author,
        "text": text,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    messages.append(message)
    MESSAGES_FILE.write_text(json.dumps(messages, indent=2))
    return message
