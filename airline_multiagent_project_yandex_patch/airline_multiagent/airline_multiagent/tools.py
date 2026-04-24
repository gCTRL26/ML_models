from __future__ import annotations

import hashlib
import json
from typing import Optional
import datetime
from langchain.tools import tool
from langgraph.types import interrupt

from airline_multiagent.data.mock_data import DB
from airline_multiagent.memory_store import load_profile, update_profile
from airline_multiagent.schemas import (
    BookingRequest,
    PassengerProfileUpdate,
    UpdateBookingRequest,
)


def resolve_date(date_str: str) -> str:
    """Resolve relative date expressions to YYYY-MM-DD.

    Handles: 'tomorrow', 'today', 'next week', 'next monday' … 'next sunday',
    and passes through already-formatted YYYY-MM-DD strings unchanged.
    """
    s = date_str.strip().lower()
    today = datetime.date.today()
    if s == "today":
        return today.isoformat()
    if s == "tomorrow":
        return (today + datetime.timedelta(days=1)).isoformat()
    if s == "next week":
        return (today + datetime.timedelta(days=7)).isoformat()
    weekdays = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
    for prefix in ("next ", ""):
        for i, name in enumerate(weekdays):
            if s == prefix + name:
                days_ahead = (i - today.weekday()) % 7 or 7
                return (today + datetime.timedelta(days=days_ahead)).isoformat()
    return date_str


@tool
def search_flights(origin: str, destination: str, date: str) -> str:
    """Search flights by route and date. Returns matching flights sorted by price."""
    resolved = resolve_date(date)
    results = [
        flight
        for flight in DB["flights"]
        if origin.lower().strip() in flight["origin"].lower()
        and destination.lower().strip() in flight["destination"].lower()
        and flight["date"] == resolved
        and flight["seats_left"] > 0
    ]
    results = sorted(results, key=lambda f: (f["price"], f["departure_time"]))
    if not results:
        return json.dumps(
            {"status": "no_results", "message": "No flights found."}, ensure_ascii=False
        )
    return json.dumps(results, ensure_ascii=False)


@tool
def get_flight_details(flight_id: str) -> str:
    """Get detailed information for a specific flight by flight_id."""
    for flight in DB["flights"]:
        if flight["flight_id"].upper() == flight_id.upper():
            return json.dumps(
                {"status": "success", "flight": flight}, ensure_ascii=False
            )
    return json.dumps(
        {"status": "error", "message": f"Flight {flight_id} not found"},
        ensure_ascii=False,
    )


@tool
def get_booking(booking_id: str) -> str:
    """Get booking details by booking_id."""
    booking = DB["bookings"].get(booking_id)
    if booking:
        return json.dumps({"status": "success", "booking": booking}, ensure_ascii=False)
    return json.dumps(
        {"status": "error", "message": f"Booking {booking_id} not found"},
        ensure_ascii=False,
    )


@tool
def lookup_policy(query: str) -> str:
    """Look up airline policies such as baggage, rebooking, cancellation, pets and refunds."""
    q = query.lower()
    scored = []
    for key, value in DB["policies"].items():
        score = sum(1 for token in key.split() if token in q)
        score += sum(
            1 for token in value["content"].lower().split() if token.strip(".,") in q
        )
        scored.append((score, key, value))
    best_score, key, value = sorted(scored, reverse=True)[0]
    if best_score == 0:
        if any(word in q for word in ["bag", "luggage", "kg", "baggage"]):
            key, value = "baggage", DB["policies"]["baggage"]
        elif any(word in q for word in ["refund", "cancel", "money back"]):
            key, value = "cancellation", DB["policies"]["cancellation"]
        elif any(word in q for word in ["rebook", "move", "change", "later flight"]):
            key, value = "rebooking", DB["policies"]["rebooking"]
        elif any(word in q for word in ["pet", "cat", "dog", "animal"]):
            key, value = "pets", DB["policies"]["pets"]
        else:
            return json.dumps(
                {"status": "error", "message": "No matching policy found"},
                ensure_ascii=False,
            )
    return json.dumps(
        {"status": "success", "policy_type": key, **value}, ensure_ascii=False
    )


@tool(args_schema=PassengerProfileUpdate)
def update_passenger_profile(key: str, value: str) -> str:
    """Update a passenger profile field used by the assistant in future turns."""
    profile = update_profile(key, value)
    return json.dumps({"status": "success", "profile": profile}, ensure_ascii=False)


@tool
def get_passenger_profile() -> str:
    """Get the persistent passenger profile with preferences and saved details."""
    return json.dumps(load_profile(), ensure_ascii=False)


@tool(args_schema=BookingRequest)
def book_flight(
    flight_id: str,
    passenger_name: str,
    email: str,
    passport: str,
    seat_preference: Optional[str] = None,
) -> str:
    """Book a flight for a passenger. Uses human approval before writing to the DB."""
    flight = next((f for f in DB["flights"] if f["flight_id"] == flight_id), None)
    if not flight:
        return json.dumps(
            {"status": "error", "message": f"Unknown flight {flight_id}"},
            ensure_ascii=False,
        )

    approval = interrupt(
        {
            "action": "book_flight",
            "flight_id": flight_id,
            "passenger_name": passenger_name,
            "email": email,
            "passport": passport,
            "seat_preference": seat_preference,
            "flight_details": {
                "origin": flight["origin"],
                "destination": flight["destination"],
                "date": flight["date"],
                "departure_time": flight["departure_time"],
                "fare_class": flight["fare_class"],
                "price": flight["price"],
            },
        }
    )

    if approval != "approved":
        return json.dumps(
            {
                "status": "cancelled",
                "message": f"Booking cancelled by operator: {approval}",
            },
            ensure_ascii=False,
        )

    ref = "BK-" + hashlib.md5(f"{flight_id}:{email}".encode()).hexdigest()[:6].upper()
    DB["bookings"][ref] = {
        "booking_id": ref,
        "passenger_name": passenger_name,
        "flight_id": flight_id,
        "origin": flight["origin"],
        "destination": flight["destination"],
        "date": flight["date"],
        "fare_class": flight["fare_class"],
        "status": "confirmed",
        "price": flight["price"],
        "email": email,
        "passport": passport,
        "seat_preference": seat_preference or "unspecified",
    }
    return json.dumps(
        {"status": "success", "booking": DB["bookings"][ref]}, ensure_ascii=False
    )


@tool(args_schema=UpdateBookingRequest)
def update_booking(booking_id: str, new_flight_id: str, new_date: str) -> str:
    """Rebook an existing booking to a new flight/date. Requires human approval."""
    booking = DB["bookings"].get(booking_id)
    if not booking:
        return json.dumps(
            {"status": "error", "message": f"Booking {booking_id} not found"},
            ensure_ascii=False,
        )

    new_flight = next(
        (
            f
            for f in DB["flights"]
            if f["flight_id"].upper() == new_flight_id.upper() and f["date"] == new_date
        ),
        None,
    )
    if not new_flight:
        return json.dumps(
            {
                "status": "error",
                "message": f"Flight {new_flight_id} on {new_date} not found",
            },
            ensure_ascii=False,
        )

    if booking["fare_class"] != new_flight["fare_class"]:
        return json.dumps(
            {
                "status": "error",
                "message": (
                    f"Class mismatch: booking is {booking['fare_class']}, new flight is {new_flight['fare_class']}. "
                    "Policy: same cabin only for self-service rebooking."
                ),
            },
            ensure_ascii=False,
        )

    fee = 50 if booking["date"] != new_date else 0
    fare_difference = max(0, new_flight["price"] - booking["price"])
    approval = interrupt(
        {
            "action": "update_booking",
            "booking_id": booking_id,
            "new_flight_id": new_flight_id,
            "new_date": new_date,
            "fee": fee,
            "fare_difference": fare_difference,
            "total_incremental_cost": fee + fare_difference,
        }
    )
    if approval != "approved":
        return json.dumps(
            {
                "status": "cancelled",
                "message": f"Rebooking cancelled by operator: {approval}",
            },
            ensure_ascii=False,
        )

    booking.update(
        {
            "flight_id": new_flight["flight_id"],
            "date": new_flight["date"],
            "status": "rebooked",
            "price": new_flight["price"],
        }
    )
    return json.dumps(
        {
            "status": "success",
            "updated_booking": booking,
            "fee": fee,
            "fare_difference": fare_difference,
            "total_incremental_cost": fee + fare_difference,
        },
        ensure_ascii=False,
    )


@tool
def cancel_booking(booking_id: str) -> str:
    """Cancel an existing booking after human approval."""
    booking = DB["bookings"].get(booking_id)
    if not booking:
        return json.dumps(
            {"status": "error", "message": f"Booking {booking_id} not found"},
            ensure_ascii=False,
        )

    approval = interrupt(
        {
            "action": "cancel_booking",
            "booking_id": booking_id,
            "current_status": booking["status"],
        }
    )
    if approval != "approved":
        return json.dumps(
            {
                "status": "cancelled",
                "message": f"Cancellation aborted by operator: {approval}",
            },
            ensure_ascii=False,
        )

    booking["status"] = "cancelled"
    return json.dumps({"status": "success", "booking": booking}, ensure_ascii=False)


FLIGHT_TOOLS = [search_flights, get_flight_details]
POLICY_TOOLS = [lookup_policy]
BOOKING_TOOLS = [get_booking, update_booking, cancel_booking, book_flight]
PROFILE_TOOLS = [get_passenger_profile, update_passenger_profile]
ALL_TOOLS = FLIGHT_TOOLS + POLICY_TOOLS + BOOKING_TOOLS + PROFILE_TOOLS
