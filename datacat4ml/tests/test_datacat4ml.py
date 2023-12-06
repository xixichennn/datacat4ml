"""
Unit and regression test for the datacat4ml package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import datacat4ml


def test_datacat4ml_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "datacat4ml" in sys.modules
