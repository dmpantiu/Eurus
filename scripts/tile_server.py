#!/usr/bin/env python3
"""
Eurus Tile Server — xpublish-tiles launcher
=============================================
Standalone tile server for interactive map rendering of ERA5 data.

Usage:
    python scripts/tile_server.py                           # Default: air tutorial on port 8080
    python scripts/tile_server.py --port 9090               # Custom port
    python scripts/tile_server.py --dataset air              # xarray tutorial dataset

Requires:
    pip install eurus[tiles]   OR   pip install xpublish-tiles
"""

import argparse
import os
import sys
from pathlib import Path

# Load .env for ARRAYLAKE_API_KEY
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def check_xpublish_tiles():
    """Check if xpublish-tiles is installed."""
    try:
        import xpublish_tiles  # noqa: F401
        return True
    except ImportError:
        return False


def load_dataset(name: str):
    """Load a dataset by name."""
    import xarray as xr

    if name == "air":
        ds = xr.tutorial.open_dataset("air_temperature")
        ds.attrs["_xpublish_id"] = "air_temperature"
        return ds
    elif name == "ersstv5":
        ds = xr.tutorial.open_dataset("ersstv5")
        ds.attrs["_xpublish_id"] = "ersstv5"
        return ds
    elif name == "rasm":
        ds = xr.tutorial.open_dataset("rasm")
        ds.attrs["_xpublish_id"] = "rasm"
        return ds
    elif name.startswith("zarr://"):
        import zarr
        ds = xr.open_zarr(name[7:])
        ds.attrs["_xpublish_id"] = name[7:]
        return ds
    else:
        # Default: air temperature
        ds = xr.tutorial.open_dataset("air_temperature")
        ds.attrs["_xpublish_id"] = "air_temperature"
        return ds


def main():
    parser = argparse.ArgumentParser(
        description="Eurus Tile Server — interactive map tile rendering"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port to serve tiles on (default: 8080)"
    )
    parser.add_argument(
        "--dataset", type=str, default="air",
        help="Dataset to serve: air (default), ersstv5, rasm, or zarr:///path"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    args = parser.parse_args()

    # Check installation
    if not check_xpublish_tiles():
        print("❌ xpublish-tiles is not installed.")
        print("   Install with: pip install xpublish-tiles")
        sys.exit(1)

    import xpublish
    from xpublish_tiles.xpublish.tiles import TilesPlugin

    # Load dataset
    print(f"📦 Loading dataset: {args.dataset}...")
    ds = load_dataset(args.dataset)
    print(f"   Variables: {list(ds.data_vars)}")
    print(f"   Dimensions: {dict(ds.dims)}")

    # Create xpublish rest server
    rest = xpublish.Rest({args.dataset: ds}, plugins={"tiles": TilesPlugin()})

    print(f"""
╔══════════════════════════════════════════════════════╗
║          Eurus Tile Server                           ║
║          powered by xpublish-tiles                   ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║   Tiles:   http://localhost:{args.port}/datasets/{args.dataset}/tiles/  ║
║   Docs:    http://localhost:{args.port}/docs                  ║
║                                                      ║
║   Dataset: {args.dataset:<40s}  ║
║                                                      ║
║   Example tile URL:                                  ║
║   /datasets/{args.dataset}/tiles/WebMercatorQuad/2/1/1     ║
║     ?variables=air&colorscalerange=240,310            ║
║     &style=raster/viridis                             ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

    # Run the server
    rest.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
