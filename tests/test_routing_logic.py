"""
Direct Test for Pure Maritime Routing
=====================================
Tests that the tool calculates a route and returns waypoints.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vostok.tools.routing import calculate_maritime_route, HAS_ROUTING_DEPS

def test_routing_direct():
    print(f"Routing Dependencies Installed: {HAS_ROUTING_DEPS}")
    
    if not HAS_ROUTING_DEPS:
        print("Skipping test.")
        return

    # Bremerhaven -> New York
    origin = (53.5, 8.5)
    dest = (40.7, -74.0)
    
    print(f"\nCalculating route: {origin} -> {dest}")
    
    result = calculate_maritime_route(
        origin_lat=origin[0], origin_lon=origin[1],
        dest_lat=dest[0], dest_lon=dest[1],
        month=11,
        speed_knots=20.0
    )
    
    print("\n--- TOOL OUTPUT ---")
    print(result)
    print("-------------------")
    
    assert "MARITIME ROUTE CALCULATED" in result
    assert "WAYPOINTS" in result
    assert "Recommended Bounding Box" in result

if __name__ == "__main__":
    test_routing_direct()