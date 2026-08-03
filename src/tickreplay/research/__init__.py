"""Research toolkit: measurement and overfitting protection.

The engine answers "what would this strategy have made"; this package answers the
harder question — "is that number real". Excursion and barrier labels for
measurement, purged walk-forward splits, overlap-thinned significance tests, an
append-only registry with cumulative false-discovery control, and an online
changepoint detector.
"""
from . import bocpd, labels, registry, stats, validation

__all__ = ["bocpd", "labels", "registry", "stats", "validation"]
