from opsdesk.models import Incident, Priority
from opsdesk.routing import route_incident


def test_checkout_outage_routes_to_payments():
    incident = Incident(
        incident_id="INC-1042",
        title="Checkout unavailable for web customers",
        description="Payment checkout outage affecting order placement.",
        service="checkout",
        customer_count=1400,
        tags={"outage"},
    )
    assert route_incident(incident) == ("incident-command", Priority.CRITICAL)


def test_security_incident_routes_to_security():
    incident = Incident(
        incident_id="INC-1043",
        title="Credential exposure in checkout logs",
        description="Possible data loss in payment traces.",
        service="checkout",
        customer_count=3,
        tags={"security"},
    )
    assert route_incident(incident) == ("security-response", Priority.CRITICAL)


def test_unknown_service_uses_general_support():
    incident = Incident("INC-1044", "Question", "How do I export data?", "billing")
    assert route_incident(incident) == ("general-support", Priority.LOW)
