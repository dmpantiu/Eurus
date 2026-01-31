"""
Superb Python REPL Tool
=======================
A persistent Python execution environment for the agent.
Supports state preservation, plotting, and data analysis.
"""

import sys
import io
import gc
import contextlib
import traceback
import matplotlib
# Force non-interactive backend to prevent crashes on headless servers
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Pre-import common scientific libraries for convenience
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# Memory monitoring
def _get_memory_mb():
    """Get current process memory usage in MB."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On macOS, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == 'darwin':
            return usage / (1024 * 1024)
        else:
            return usage / 1024
    except:
        return 0

MAX_MEMORY_MB = 4000  # 4GB limit

# Import terminal display for inline images
try:
    from vostok.terminal_display import display_image
    HAS_TERMINAL_DISPLAY = True
except ImportError:
    HAS_TERMINAL_DISPLAY = False
    display_image = None

# Create a wrapper for plt.savefig that auto-displays
_original_savefig = plt.savefig

def _savefig_with_display(*args, **kwargs):
    """Save figure and automatically display in terminal."""
    result = _original_savefig(*args, **kwargs)
    
    # Try to display the saved image
    if args:
        filepath = args[0]
        if isinstance(filepath, str) and filepath.endswith(('.png', '.jpg', '.jpeg')):
            # 1. Notify LLM (captured stdout)
            print(f"Plot saved to: {filepath}")
            
            # 2. Display to User (real stdout)
            current_stdout = sys.stdout
            try:
                sys.stdout = sys.__stdout__
                print(f"\n📊 Saved: {filepath}")
                
                # Try terminal display first
                displayed = False
                if HAS_TERMINAL_DISPLAY:
                    displayed = display_image(filepath)
                
                # If terminal doesn't support images AND we're on macOS, open in Preview
                if not displayed:
                    import subprocess
                    import platform
                    if platform.system() == 'Darwin':  # macOS
                        try:
                            subprocess.Popen(['open', filepath], 
                                           stdout=subprocess.DEVNULL, 
                                           stderr=subprocess.DEVNULL)
                            print(f"📂 Opened in Preview")
                        except:
                            print(f"   (Open manually to view)")
                        
            except Exception:
                pass
            finally:
                sys.stdout = current_stdout
    
    return result

# Monkey-patch plt.savefig
plt.savefig = _savefig_with_display


class PythonREPLInput(BaseModel):
    code: str = Field(description="The Python code to execute.")

class SuperbPythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = (
        "A Python REPL for data analysis. CRITICAL MEMORY RULES: "
        "1) NEVER use .load() or .compute() on large datasets - use lazy evaluation. "
        "2) For multi-year hourly data: resample to daily/monthly FIRST with .resample('D').mean(). "
        "3) Use .sel() to subset data before operations. "
        "4) For plots: aggregate data BEFORE plotting (e.g., annual means). "
        "5) Close figures with plt.close() after saving. "
        "Variables persist between calls. Save plots to ./data/plots/."
    )
    args_schema: Type[BaseModel] = PythonREPLInput
    globals_dict: Dict = Field(default_factory=dict, exclude=True)
    working_dir: str = "."

    def __init__(self, working_dir: str = ".", **kwargs):
        super().__init__(**kwargs)
        self.working_dir = working_dir
        # Initialize globals with common libraries (no os for security)
        self.globals_dict = {
            "pd": pd,
            "np": np,
            "xr": xr,
            "plt": plt,
            "datetime": datetime,
            "timedelta": timedelta,
        }


    def _run(self, code: str) -> str:
        """Execute the python code and return the output."""
        import threading
        
        # Security: Block dangerous operations
        dangerous_patterns = [
            "import os", "from os", "os.system", "os.popen", "os.remove", "os.rmdir",
            "import subprocess", "from subprocess", "subprocess.",
            "import shutil", "shutil.rmtree",
            "__import__", "eval(", "exec(",
            "open(", "file(",  # Block file writes (reading existing data is OK via xr/pd)
            "import socket", "socket.",
            "import requests", "requests.",
        ]
        
        code_lower = code.lower().replace(" ", "")
        for pattern in dangerous_patterns:
            if pattern.lower().replace(" ", "") in code_lower:
                return f"Security Error: '{pattern}' is not allowed. This REPL is for data analysis only."
        
        # Execute with timeout using threading
        result_container = {"output": None, "error": None}
        
        def execute_code():
            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output
            
            try:
                # Try to compile as an expression first (like a real REPL)
                try:
                    compiled = compile(code, '<repl>', 'eval')
                    result = eval(compiled, self.globals_dict)
                    output = redirected_output.getvalue()
                    if result is not None:
                        output += repr(result)
                    result_container["output"] = output.strip() if output.strip() else repr(result) if result is not None else "(No output)"
                except SyntaxError:
                    # Not an expression, execute as statements
                    exec(code, self.globals_dict)
                    output = redirected_output.getvalue()
                    
                    if not output.strip():
                        result_container["output"] = "(Executed successfully. Use print() to see results.)"
                    else:
                        result_container["output"] = output.strip()
                    
            except Exception as e:
                result_container["error"] = f"Error: {str(e)}\n{traceback.format_exc()}"
                
            finally:
                sys.stdout = old_stdout
                plt.close('all')
                gc.collect()
        
        # Run in thread with 60-second timeout
        exec_thread = threading.Thread(target=execute_code)
        exec_thread.start()
        exec_thread.join(timeout=60)
        
        if exec_thread.is_alive():
            # Thread is still running after timeout
            return "TIMEOUT ERROR: Execution exceeded 60 seconds. TIP: Resample data to daily/monthly before plotting (e.g., ds.resample(time='D').mean())."
        
        if result_container["error"]:
            return result_container["error"]
        
        return result_container["output"] or "(No output)"

    async def _arun(self, code: str) -> str:
        """Use the tool asynchronously."""
        return self._run(code)
