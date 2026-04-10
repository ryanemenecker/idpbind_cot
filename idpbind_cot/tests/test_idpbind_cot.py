"""
Unit and regression test for the idpbind_cot package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import idpbind_cot


def test_idpbind_cot_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "idpbind_cot" in sys.modules
