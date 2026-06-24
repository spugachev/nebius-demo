"""Mock implementations of the eval tools.

Used by compare.py for the `executable_success_rate` metric: a predicted tool call
is "executable" if its function exists here and the call runs without raising
(i.e. required arguments are present and well-typed). Returns are canned — we test
call *correctness*, not real side effects.
"""
from __future__ import annotations


def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": 18, "conditions": "partly cloudy"}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    return {"amount": amount, "from": from_currency, "to": to_currency, "result": round(amount * 0.92, 2)}


def get_stock_price(symbol: str) -> dict:
    return {"symbol": symbol, "price": 123.45, "currency": "USD"}


def set_reminder(text: str, datetime: str) -> dict:
    return {"status": "scheduled", "text": text, "datetime": datetime}


def translate_text(text: str, target_language: str) -> dict:
    return {"translation": f"<{target_language}> {text}"}


def search_inventory(sku: str) -> dict:
    return {"sku": sku, "in_stock": 42}


def send_email(to: str, subject: str, body: str) -> dict:
    return {"status": "sent", "to": to, "subject": subject}


def schedule_meeting(title: str, attendee: str, start: str, duration_minutes: int) -> dict:
    return {"status": "booked", "title": title, "attendee": attendee, "start": start, "duration_minutes": duration_minutes}


def book_flight(origin: str, destination: str, date: str, passengers: int, cabin: str = "economy") -> dict:
    return {"status": "booked", "origin": origin, "destination": destination, "date": date, "passengers": passengers, "cabin": cabin}


def create_invoice(customer: str, amount: float, currency: str, due_date: str) -> dict:
    return {"invoice_id": "INV-1001", "customer": customer, "amount": amount, "currency": currency, "due_date": due_date}


def update_crm_contact(email: str, status: str = None, company: str = None) -> dict:
    return {"status": "updated", "email": email, "fields": {"status": status, "company": company}}


def create_ticket(customer: str, category: str, priority: str = "medium") -> dict:
    return {"ticket_id": "TIC-5005", "customer": customer, "category": category, "priority": priority}


def create_employee_onboarding(name: str, department: str, start_date: str) -> dict:
    return {"workflow_id": "ONB-300", "name": name, "department": department, "start_date": start_date}


def issue_refund(order_id: str, amount: float, currency: str) -> dict:
    return {"status": "refunded", "order_id": order_id, "amount": amount, "currency": currency}


def provision_vm(name: str, vcpus: int, ram_gb: int, region: str) -> dict:
    return {"status": "provisioning", "name": name, "vcpus": vcpus, "ram_gb": ram_gb, "region": region}


MOCK_TOOLS = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
    "get_stock_price": get_stock_price,
    "set_reminder": set_reminder,
    "translate_text": translate_text,
    "search_inventory": search_inventory,
    "send_email": send_email,
    "schedule_meeting": schedule_meeting,
    "book_flight": book_flight,
    "create_invoice": create_invoice,
    "update_crm_contact": update_crm_contact,
    "create_ticket": create_ticket,
    "create_employee_onboarding": create_employee_onboarding,
    "issue_refund": issue_refund,
    "provision_vm": provision_vm,
}
