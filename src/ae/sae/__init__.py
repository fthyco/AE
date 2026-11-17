"""SAE: non-executing Static Architecture Extractor."""
from .api import extract, inspect_bundle, validate_bundle

__all__ = ["extract", "validate_bundle", "inspect_bundle"]
