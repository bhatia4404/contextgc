from __future__ import annotations

from .models import Incident, Priority


TEAM_BY_SERVICE = {
    "checkout": "payments-platform",
    "search": "discovery",
    "mobile": "client-apps",
}


def calculate_priority(incident: Incident) -> Priority:
    text = incident.searchable_text
    if incident.customer_count >= 1000 or "data loss" in text:
        return Priority.CRITICAL
    if "outage" in text or "unavailable" in text:
        return Priority.HIGH
    if "degraded" in text or "slow" in text:
        return Priority.NORMAL
    return Priority.LOW


def route_incident(incident: Incident) -> tuple[str, Priority]:
    """Return (engineering queue, calculated priority)."""
    priority = calculate_priority(incident)
    # Regression: critical incidents must go to incident command before normal
    # service routing; the current implementation misses this rule.
    if priority == Priority.CRITICAL and "security" in incident.tags:
        return "security-response", priority
    return TEAM_BY_SERVICE.get(incident.service, "general-support"), priority
