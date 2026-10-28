from opsdesk.models import Incident
from opsdesk.summary import incident_summary


def test_summary_contains_operational_fields():
    incident = Incident("INC-1050", "Search slow", "p95 latency increased", "search")
    summary = incident_summary(incident)
    assert "INC-1050" in summary
    assert "queue=discovery" in summary
    assert "NORMAL" in summary
