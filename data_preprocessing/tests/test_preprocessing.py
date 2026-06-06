import pytest


# Test add_fingerprints
def test_add_fingerprints(comp_metadata):
    if comp_metadata.isna.any:
        