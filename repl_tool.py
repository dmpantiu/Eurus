"""
Superb Python REPL Tool (Enhanced)
===================================
A persistent Jupyter kernel with advanced visualization and memory.

Features:
- Persistent state between calls
- Automatic plot saving and display
- Enhanced output capture (stdout, stderr, images)
- Pre-loaded scientific computing libraries
- Analysis tracking in memory
"""

import os
import sys
import re
import time
import queue
import base64
import logging
import atexit
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

from config import DATA_DIR, PLOTS_DIR, CONFIG

# Ensure Jupyter is installed
try:
    from jupyter_client import KernelManager
    from jupyter_client.client import KernelClient
except ImportError:
    raise ImportError("Missing 'jupyter_client'. Run: pip install jupyter_client ipykernel")

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# JUPYTER KERNEL EXECUTOR
# ============================================================================

class JupyterKernelExecutor:
    """
    Manages a persistent Jupyter kernel with state preservation.

    Features:
    - Variables and imports preserved between calls
    - Automatic plot capture and saving
    - Enhanced output handling
    - Timeout management with kernel interrupt
    - Optional plot callback for web interface
    """

    def __init__(
        self,
        working_dir: str = None,
        timeout: float = 300.0,
        on_plot_callback: Optional[Any] = None
    ):
        self._working_dir = working_dir or os.getcwd()
        self._timeout = timeout
        self._plots_dir = Path(PLOTS_DIR)
        self._plots_dir.mkdir(parents=True, exist_ok=True)

        self._plot_counter = 0
        self._captured_plots: List[str] = []

        # Callback for web interface plot streaming
        self._on_plot_callback = on_plot_callback

        self.km = KernelManager(kernel_name='python3')
        self.kc: Optional[KernelClient] = None

        self._start_kernel()

    def _start_kernel(self):
        """Start the Jupyter kernel with pre-loaded libraries."""
        logger.info("Starting Jupyter kernel...")

        # Start kernel in working directory
        if os.path.exists(self._working_dir):
            self.km.start_kernel(cwd=self._working_dir)
        else:
            self.km.start_kernel()

        self.kc = self.km.client()
        self.kc.start_channels()

        try:
            self.kc.wait_for_ready(timeout=60)
            logger.info("Kernel ready.")
        except RuntimeError as e:
            self.km.shutdown_kernel()
            raise RuntimeError(f"Kernel failed to start: {e}")

        # Initialize the kernel with essential imports only
        init_code = f'''
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Core scientific stack
import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta

# Visualization - use inline backend for automatic plot capture
%matplotlib inline
import matplotlib.pyplot as plt

# Set matplotlib defaults
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['font.size'] = 11

# Useful directories
DATA_DIR = "{DATA_DIR}"
PLOTS_DIR = "{PLOTS_DIR}"

print("Kernel Ready!")
print("Pre-loaded: pandas (pd), numpy (np), xarray (xr), matplotlib.pyplot (plt)")
'''
        # Execute initialization
        result = self.execute(init_code)
        logger.info("Kernel initialized with scientific libraries")

    def execute(self, code: str, timeout: float = None) -> Dict[str, Any]:
        """
        Execute Python code in the kernel.

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds

        Returns:
            Dict with 'output', 'errors', 'plots', 'success'
        """
        if not self.kc:
            return {"output": "Error: Kernel not connected.", "errors": [], "plots": [], "success": False}

        timeout = timeout or self._timeout

        # Clean code (remove markdown code blocks if present)
        code = re.sub(r"^(\s|`)*(?i:python)?\s*", "", code)
        code = re.sub(r"(\s|`)*$", "", code)

        msg_id = self.kc.execute(code)

        # Collect outputs
        stdout_parts = []
        stderr_parts = []
        error_parts = []
        display_data = []
        plots = []

        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self.km.interrupt_kernel()
                return {
                    "output": "Error: Execution timed out after {:.1f}s".format(timeout),
                    "errors": ["Timeout"],
                    "plots": [],
                    "success": False
                }

            try:
                msg = self.kc.get_iopub_msg(timeout=0.1)
            except queue.Empty:
                continue

            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            msg_type = msg['msg_type']
            content = msg['content']

            if msg_type == 'status' and content['execution_state'] == 'idle':
                break

            elif msg_type == 'stream':
                if content['name'] == 'stdout':
                    stdout_parts.append(content['text'])
                elif content['name'] == 'stderr':
                    stderr_parts.append(content['text'])

            elif msg_type in ('execute_result', 'display_data'):
                data = content.get('data', {})

                # Handle images
                if 'image/png' in data:
                    plot_path = self._save_captured_image(data['image/png'])
                    if plot_path:
                        plots.append(plot_path)

                # Handle text output
                if 'text/plain' in data:
                    text = data['text/plain']
                    display_data.append(text)

            elif msg_type == 'error':
                # Strip ANSI codes for readability
                traceback = "\n".join(content['traceback'])
                traceback = re.sub(r'\x1b\[[0-9;]*m', '', traceback)
                error_parts.append(f"{content['ename']}: {content['evalue']}\n{traceback}")

        # Build output string
        output_parts = []

        if stdout_parts:
            output_parts.append("".join(stdout_parts))

        if display_data:
            output_parts.append("\n".join(display_data))

        if stderr_parts:
            stderr_text = "".join(stderr_parts)
            # Filter out common non-critical warnings
            if stderr_text.strip() and 'FutureWarning' not in stderr_text:
                output_parts.append(f"\n[STDERR]:\n{stderr_text}")

        if error_parts:
            output_parts.append(f"\n[ERROR]:\n" + "\n".join(error_parts))

        if plots:
            output_parts.append(f"\n[PLOTS SAVED]: {', '.join(plots)}")
            self._captured_plots.extend(plots)

        output = "\n".join(output_parts).strip()

        if not output:
            output = "[Code executed successfully. No output.]"

        return {
            "output": output,
            "errors": error_parts,
            "plots": plots,
            "success": len(error_parts) == 0
        }

    def set_plot_callback(self, callback: Any):
        """Set callback for plot events (used by web interface)."""
        self._on_plot_callback = callback

    def _save_captured_image(self, base64_data: str) -> str:
        """Save a base64-encoded image to the plots directory."""
        self._plot_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}_{self._plot_counter}.png"
        filepath = self._plots_dir / filename

        try:
            image_data = base64.b64decode(base64_data)
            with open(filepath, 'wb') as f:
                f.write(image_data)
            logger.info(f"Plot saved: {filepath}")

            # Call the plot callback if registered (for web interface)
            if self._on_plot_callback:
                try:
                    self._on_plot_callback(base64_data, str(filepath))
                except Exception as e:
                    logger.error(f"Plot callback error: {e}")

            # Display in terminal if supported
            self._display_image_in_terminal(base64_data)

            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")
            return ""

    def _display_image_in_terminal(self, base64_data: str):
        """Display image inline in supported terminals (iTerm2, VSCode)."""
        # iTerm2 / VSCode inline image protocol
        # \033]1337;File=inline=1:{base64}\a
        try:
            # Check if we are likely in a supported terminal
            term_program = os.environ.get("TERM_PROGRAM", "")
            term = os.environ.get("TERM", "")
            
            supported = False
            if "iTerm.app" in term_program:
                supported = True
            elif "vscode" in term_program:
                supported = True
            elif "xterm-kitty" in term:
                # Kitty has a different protocol, but for now we focus on iTerm2/VSCode
                pass
                
            if supported:
                # Print the escape code to stdout
                # We use width=auto to let the terminal decide size, but you can enforce width=100% etc.
                # For VSCode, sometimes explicitly setting width helps.
                sys.stdout.write(f"\033]1337;File=inline=1;width=auto;preserveAspectRatio=1:{base64_data}\a\n")
                sys.stdout.flush()
        except Exception as e:
            logger.warning(f"Failed to display image in terminal: {e}")

    def get_captured_plots(self) -> List[str]:
        """Get list of all captured plot paths."""
        return self._captured_plots.copy()

    def clear_captured_plots(self):
        """Clear the list of captured plots."""
        self._captured_plots.clear()

    def close(self):
        """Shutdown the kernel."""
        logger.info("Shutting down kernel...")
        if self.kc:
            self.kc.stop_channels()
        if self.km and self.km.is_alive():
            self.km.shutdown_kernel(now=True)
        logger.info("Kernel shutdown complete.")


# ============================================================================
# LANGCHAIN TOOL WRAPPER
# ============================================================================

class PythonREPLInput(BaseModel):
    """Input schema for Python REPL."""
    code: str = Field(
        description=(
            "Python code to execute. Variables persist between calls.\n"
            "Pre-loaded: pandas (pd), numpy (np), xarray (xr), matplotlib.pyplot (plt)\n"
            "Load data with: ds = xr.open_dataset('path.zarr', engine='zarr')\n"
            "Always use print() to show results."
        )
    )


class SuperbPythonREPLTool(BaseTool):
    """
    A persistent Python environment for data analysis and visualization.

    Features:
    - State persists between calls (variables, imports)
    - Pre-loaded: pandas, numpy, xarray, matplotlib
    - Plots are captured and saved to ./data/plots/
    """

    name: str = "python_repl"
    description: str = (
        "A persistent Python Jupyter environment for analysis and visualization.\n\n"
        "PRE-LOADED:\n"
        "- pandas (pd), numpy (np), xarray (xr)\n"
        "- matplotlib.pyplot (plt) - headless mode\n"
        "- datetime, timedelta\n\n"
        "USAGE:\n"
        "1. Load data: ds = xr.open_dataset('path.zarr', engine='zarr')\n"
        "2. Analyze: print(ds.mean(dim='time'))\n"
        "3. Plot: plt.figure(); ds['sst'].plot(); plt.savefig('./data/plots/sst.png'); plt.close()\n\n"
        "Variables PERSIST between calls. Always print() results!"
    )

    args_schema: type[BaseModel] = PythonREPLInput

    _executor: JupyterKernelExecutor = PrivateAttr()

    def __init__(self, working_dir: str = None, **kwargs):
        super().__init__(**kwargs)
        self._executor = JupyterKernelExecutor(
            working_dir=working_dir,
            timeout=CONFIG.kernel_timeout
        )
        # Register cleanup on exit
        atexit.register(self._executor.close)

    def _run(self, code: str) -> str:
        """Execute Python code."""
        # Display code being executed
        print(f"\n{'='*60}")
        print("EXECUTING CODE:")
        print("-" * 60)
        print(code)
        print("=" * 60)

        # Execute
        result = self._executor.execute(code)

        # Format output
        output = result["output"]

        # Add plot info if any
        if result["plots"]:
            print(f"\nPlots generated: {len(result['plots'])}")
            for plot_path in result["plots"]:
                print(f"  - {plot_path}")

        return output

    async def _arun(self, code: str) -> str:
        """Async execution (delegates to sync)."""
        return self._run(code)

    def close(self):
        """Close the kernel."""
        self._executor.close()

    def get_plots(self) -> List[str]:
        """Get all captured plot paths."""
        return self._executor.get_captured_plots()


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing SuperbPythonREPLTool...")

    tool = SuperbPythonREPLTool(working_dir=os.getcwd())

    # Test basic execution
    result = tool._run("print('Hello from REPL!')")
    print(f"Result: {result}")

    # Test data manipulation
    result = tool._run("""
import numpy as np
data = np.random.randn(100)
print(f"Mean: {data.mean():.4f}")
print(f"Std: {data.std():.4f}")
""")
    print(f"Result: {result}")

    # Cleanup
    tool.close()
    print("Test complete!")
