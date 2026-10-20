"""Identification module for salamander re-identification."""

from app.identification.embedder import SalamanderEmbedder
from app.identification.feature_store import FeatureStore
from app.identification.verifier import SalamanderVerifier

__all__ = ["FeatureStore", "SalamanderEmbedder", "SalamanderVerifier"]
