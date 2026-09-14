"""Retry helpers for flaky network calls."""

import time


def retry_with_backoff(fn, max_attempts=3, base_delay=0.1):
    """Call fn, retrying with exponential backoff if it raises."""
    for attempt in range(max_attempts):
        try:
            return fn()
        except ConnectionError:
            if attempt == max_attempts - 1:
                raise
            time.sleep(base_delay * (2**attempt))


class RetryPolicy:
    """Configures how connection retries behave for a client."""

    def __init__(self, max_attempts=3, base_delay=0.1):
        self.max_attempts = max_attempts
        self.base_delay = base_delay

    def should_retry(self, attempt, error):
        """Return True if another attempt should be made after this error."""
        return attempt < self.max_attempts and isinstance(error, ConnectionError)
