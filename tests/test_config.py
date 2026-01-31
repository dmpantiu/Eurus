import pytest
from vostok.config import ERA5_VARIABLES, get_variable_info, get_short_name

def test_variable_loading():
    """Test that ERA5 variables are loaded correctly."""
    assert "sst" in ERA5_VARIABLES
    assert "t2" in ERA5_VARIABLES
    assert "u10" in ERA5_VARIABLES
    
    sst_info = ERA5_VARIABLES["sst"]
    assert sst_info.units == "K"
    assert sst_info.short_name == "sst"

def test_get_variable_info():
    """Test helper function for retrieving variable info."""
    # Test case insensitive
    assert get_variable_info("SST") == ERA5_VARIABLES["sst"]
    assert get_variable_info("Sea_Surface_Temperature") == ERA5_VARIABLES["sst"]
    assert get_variable_info("non_existent_var") is None

def test_get_short_name():
    """Test retrieval of short names."""
    assert get_short_name("SST") == "sst"
    assert get_short_name("Sea_Surface_Temperature") == "sst"
    # Fallback to lower case input
    assert get_short_name("UNKNOWN_VAR") == "unknown_var"

def test_agent_prompt_branding():
    """Test that the system prompt contains the Vostok branding."""
    from vostok.config import AGENT_SYSTEM_PROMPT
    assert "Vostok" in AGENT_SYSTEM_PROMPT
    assert "Comrade Copernicus" not in AGENT_SYSTEM_PROMPT
    assert "PANGAEA" not in AGENT_SYSTEM_PROMPT
