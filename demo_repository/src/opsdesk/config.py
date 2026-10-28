from __future__ import annotations

import os


def get_config() -> dict[str, str]:
    return {
        "environment": os.getenv("OPSdesk_ENV", "production"),
        "default_queue": os.getenv("OPSDESK_DEFAULT_QUEUE", "general-support"),
        "on_call": os.getenv("OPSDESK_ON_CALL", "platform-primary"),
    }
