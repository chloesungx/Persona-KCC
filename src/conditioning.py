"""Persona attribute selection for the publication snapshot.

The development version queried a private database at import time. The KCC ANES
run uses all attributes and therefore needs no database connection.
"""


def get_relevant_attributes(query: str) -> list[str]:
    """Return an empty list to signal that all profile attributes are used."""
    del query
    return []
