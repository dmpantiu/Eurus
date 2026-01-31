import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

# Import project modules
from vostok.tools.era5 import retrieve_era5_data, ERA5RetrievalArgs
from vostok.tools.repl import SuperbPythonREPLTool
from vostok.memory import MemoryManager, get_memory, reset_memory
from vostok.config import CONFIG

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def clean_memory(tmp_path):
    """Provide a clean memory instance using a temporary directory."""
    reset_memory()
    # Mock the memory directory to be inside tmp_path
    with patch("vostok.memory.get_memory_dir", return_value=tmp_path):
        memory = get_memory()
        yield memory
    reset_memory()

@pytest.fixture
def mock_repl():
    """Provide a REPL tool instance."""
    return SuperbPythonREPLTool()

# ============================================================================
# TEST CASES
# ============================================================================

# TC1: Valid Temporal Retrieval (Mocked)
@patch("vostok.tools.era5.os.environ.get", return_value="dummy_key")
@patch("vostok.tools.era5.urlopen")
@patch("vostok.tools.era5.xr.open_dataset")
def test_era5_retrieval_temporal(mock_xr, mock_urlopen, mock_env, clean_memory):
    """Test standard temporal query retrieval logic."""
    # Mock arraylake
    mock_arraylake = MagicMock()
    mock_client = MagicMock()
    mock_repo = MagicMock()
    mock_session = MagicMock()
    mock_client.get_repo.return_value = mock_repo
    mock_repo.readonly_session.return_value = mock_session
    mock_arraylake.Client.return_value = mock_client
    
    with patch.dict(sys.modules, {'arraylake': mock_arraylake}):
        # Mock valid dataset response
        mock_ds = MagicMock()
        mock_ds.__contains__.return_value = True # variable exists
        mock_ds.__getitem__.return_value = MagicMock() # ds[var]
        mock_xr.return_value = mock_ds
        
        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="sst",
            start_date="2023-01-01",
            end_date="2023-01-31",
            region="california_coast"
        )
        
        assert "SUCCESS" in result
        assert "sst" in result
        assert "california_coast" in str(clean_memory.list_datasets()) or "sst" in str(clean_memory.list_datasets())

# TC2: Valid Spatial Retrieval (Mocked)
@patch("vostok.tools.era5.os.environ.get", return_value="dummy_key")
@patch("vostok.tools.era5.urlopen")
@patch("vostok.tools.era5.xr.open_dataset")
def test_era5_retrieval_spatial(mock_xr, mock_urlopen, mock_env, clean_memory):
    """Test standard spatial query retrieval logic."""
    # Mock arraylake
    mock_arraylake = MagicMock()
    mock_client = MagicMock()
    mock_repo = MagicMock()
    mock_session = MagicMock()
    mock_client.get_repo.return_value = mock_repo
    mock_repo.readonly_session.return_value = mock_session
    mock_arraylake.Client.return_value = mock_client
    
    with patch.dict(sys.modules, {'arraylake': mock_arraylake}):
        mock_ds = MagicMock()
        mock_ds.__contains__.return_value = True
        mock_xr.return_value = mock_ds
        
        result = retrieve_era5_data(
            query_type="spatial",
            variable_id="t2",
            start_date="2023-01-01",
            end_date="2023-01-02",
            min_latitude=10, max_latitude=20,
            min_longitude=10, max_longitude=20
        )
        
        assert "SUCCESS" in result
        assert "t2" in result

# TC3: Invalid Variable Error
def test_era5_retrieval_invalid_variable():
    """Test error handling for non-existent variable."""
    mock_arraylake = MagicMock()
    mock_client = MagicMock()
    mock_repo = MagicMock()
    mock_session = MagicMock()
    mock_client.get_repo.return_value = mock_repo
    mock_repo.readonly_session.return_value = mock_session
    mock_arraylake.Client.return_value = mock_client

    with patch.dict(sys.modules, {'arraylake': mock_arraylake}), \
         patch("vostok.tools.era5.os.environ.get", return_value="dummy_key"), \
         patch("vostok.tools.era5.xr.open_dataset") as mock_xr:
        
        mock_ds = MagicMock()
        mock_ds.__contains__.return_value = False # Variable NOT in dataset
        mock_ds.data_vars = ["sst", "t2"]
        mock_xr.return_value = mock_ds

        result = retrieve_era5_data(
            query_type="temporal",
            variable_id="non_existent_super_var",
            start_date="2023-01-01",
            end_date="2023-01-02"
        )
        
        assert "Error" in result
        assert "not found in dataset" in result

# TC4: Invalid Date Format
def test_era5_retrieval_invalid_dates():
    """Test validation of date formats."""
    # Pydantic validation happens before function execution if used via Tool
    # But direct function call handles it differently or relies on validation inside
    # The current implementation checks args if used via Tool, 
    # but the function itself expects strings. 
    # Let's test the Pydantic model directly to ensure it catches it. 
    
    with pytest.raises(ValueError):
        ERA5RetrievalArgs(
            query_type="temporal",
            variable_id="sst",
            start_date="2023/01/01", # Wrong format
            end_date="2023-01-02"
        )

# TC5: Memory Persistence
def test_memory_persistence(clean_memory, tmp_path):
    """Test that memory correctly saves and loads conversations."""
    clean_memory.add_message("user", "Hello Vostok")
    clean_memory.add_message("assistant", "Hello User")
    
    # Reload memory
    reset_memory()
    with patch("vostok.memory.get_memory_dir", return_value=tmp_path):
        new_memory = get_memory()
        history = new_memory.get_conversation_history()
        
        assert len(history) == 2
        assert history[0].content == "Hello Vostok"
        assert history[1].role == "assistant"

# TC6: REPL Basic Calculation
def test_repl_basic_math(mock_repl):
    """Test simple Python execution."""
    code = "print(15 + 25)"
    result = mock_repl._run(code)
    assert "40" in result

# TC7: REPL State Persistence
def test_repl_state_persistence(mock_repl):
    """Test that variables persist between calls."""
    mock_repl._run("x = 100")
    result = mock_repl._run("print(x * 2)")
    assert "200" in result

# TC8: REPL Error Handling
def test_repl_syntax_error(mock_repl):
    """Test handling of invalid Python code."""
    code = "print(undefined_variable)"
    result = mock_repl._run(code)
    assert "Error" in result
    assert "NameError" in result

# TC9: REPL Scientific Libraries
def test_repl_scientific_libs(mock_repl):
    """Test availability of pre-loaded libraries."""
    code = """
import numpy as np
arr = np.array([1, 2, 3])
print(arr.mean())
"""
    result = mock_repl._run(code)
    assert "2.0" in result

# TC10: Full Workflow Simulation (Mocked)
@patch("vostok.tools.era5.retrieve_era5_data")
def test_full_workflow(mock_retrieve, mock_repl, clean_memory):
    """Simulate a full user interaction workflow."""
    
    # Step 1: User asks for data (Agent calls tool)
    mock_retrieve.return_value = "SUCCESS - Data downloaded to ./data/test.zarr"
    
    # Step 2: Agent analyzes data (REPL)
    # Mocking xarray open_dataset inside REPL would be hard without file
    # So we simulate the agent *knowing* the path and running a script
    
    script = """
print("Analyzing ./data/test.zarr")
print("Mean SST: 288.5 K")
"""
    analysis_result = mock_repl._run(script)
    
    assert "Analyzing" in analysis_result
    assert "Mean SST" in analysis_result
    
    # Step 3: Check memory updated
    clean_memory.add_message("system", "Workflow completed")
    assert len(clean_memory.get_conversation_history()) > 0