from __future__ import annotations

from .models import Incident
from .routing import route_incident


def incident_summary(incident: Incident) -> str:
    team, priority = route_incident(incident)
    return (
        f"[{priority.value.upper()}] {incident.incident_id}: {incident.title}\n"
        f"service={incident.service} queue={team} customers={incident.customer_count}\n"
        f"{incident.description}"
    )
