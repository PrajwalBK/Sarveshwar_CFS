import json
import logging
from datetime import datetime, timezone


class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Internal event names and allowlisted scalar fields only. Exception strings
        # may contain DSNs/URLs, so only exception TYPE is emitted by callers.
        return json.dumps({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'event': record.getMessage(),
            **{k: getattr(record, k) for k in ('camera_id', 'event_id', 'error_type') if hasattr(record, k)},
        })


def configure_logging():
    logger = logging.getLogger('gate')
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
