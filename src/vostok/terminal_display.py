"""
Terminal Image Display
======================
Display images directly in the terminal using various protocols.
"""

import os
import sys
import base64
from pathlib import Path
from typing import Optional


def get_terminal_type() -> str:
    """Detect terminal type for image display."""
    term_program = os.environ.get('TERM_PROGRAM', '')
    term = os.environ.get('TERM', '')
    
    if 'iTerm' in term_program:
        return 'iterm2'
    elif 'kitty' in term.lower() or 'KITTY_WINDOW_ID' in os.environ:
        return 'kitty'
    elif 'xterm' in term or 'sixel' in term.lower():
        return 'sixel'
    else:
        return 'fallback'


def display_image_iterm2(image_path: str, width: Optional[int] = None) -> bool:
    """Display image using iTerm2's imgcat protocol."""
    try:
        path = Path(image_path)
        if not path.exists():
            return False
            
        with open(path, 'rb') as f:
            image_data = f.read()
        
        encoded = base64.b64encode(image_data).decode('ascii')
        
        # iTerm2 inline image protocol
        width_param = f";width={width}px" if width else ";width=auto"
        osc = f"\033]1337;File=inline=1{width_param}:"
        
        sys.stdout.write(osc)
        sys.stdout.write(encoded)
        sys.stdout.write("\a\n")  # Bell character terminates
        sys.stdout.flush()
        
        return True
    except Exception:
        return False


def display_image_kitty(image_path: str, width: Optional[int] = None) -> bool:
    """Display image using Kitty's graphics protocol."""
    try:
        from term_image.image import KittyImage
        
        img = KittyImage.from_file(image_path)
        if width:
            img.set_size(width=width)
        print(img)
        return True
    except Exception:
        return False


def display_image_sixel(image_path: str, width: Optional[int] = None) -> bool:
    """Display image using Sixel protocol."""
    try:
        from term_image.image import SixelImage
        
        img = SixelImage.from_file(image_path)
        if width:
            img.set_size(width=width)
        print(img)
        return True
    except Exception:
        return False


def display_image_fallback(image_path: str) -> bool:
    """Fallback: Use term-image auto-detection."""
    try:
        from term_image.image import AutoImage
        
        img = AutoImage.from_file(image_path)
        img.draw()
        return True
    except Exception:
        return False


def display_image(image_path: str, width: int = 60) -> bool:
    """
    Display an image in the terminal.
    
    Automatically detects terminal type and uses the best available method.
    
    Args:
        image_path: Path to the image file
        width: Display width in terminal columns (default 60)
        
    Returns:
        True if display succeeded, False otherwise
    """
    path = Path(image_path)
    if not path.exists():
        print(f"⚠️  Image not found: {image_path}")
        return False
    
    terminal = get_terminal_type()
    
    # Try terminal-specific method first
    if terminal == 'iterm2':
        if display_image_iterm2(image_path, width * 10):  # pixels
            return True
    elif terminal == 'kitty':
        if display_image_kitty(image_path, width):
            return True
    elif terminal == 'sixel':
        if display_image_sixel(image_path, width):
            return True
    
    # Fallback to auto-detection
    if display_image_fallback(image_path):
        return True
    
    # Final fallback: just return False, caller handles the message
    return False


def display_plot_inline(fig, filename: str = None, width: int = 60) -> str:
    """
    Save matplotlib figure and display it inline in terminal.
    
    Args:
        fig: matplotlib figure object
        filename: Optional filename, auto-generated if not provided
        width: Display width in terminal columns
        
    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Ensure plots directory exists
    plots_dir = Path("data/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}.png"
    
    # Full path
    plot_path = plots_dir / filename
    
    # Save with tight layout
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # Display in terminal
    print()
    display_image(str(plot_path), width)
    print()
    
    return str(plot_path)


# Quick test function
def test_display():
    """Test image display capabilities."""
    print(f"Terminal type detected: {get_terminal_type()}")
    print(f"Testing with a sample plot...")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x), 'b-', label='sin(x)')
    ax.plot(x, np.cos(x), 'r--', label='cos(x)')
    ax.set_title('Test Plot - Terminal Image Display')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = display_plot_inline(fig, "test_terminal_display.png")
    print(f"Plot saved to: {path}")


if __name__ == "__main__":
    test_display()
