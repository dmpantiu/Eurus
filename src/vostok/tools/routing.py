"""
Maritime Routing Tool
=====================
Strictly calculates maritime routes using global shipping lane graphs.
Does NOT perform weather analysis. Returns waypoints for the Agent to analyze.

Dependencies:
- scgraph (for maritime pathfinding)
"""

import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Any
from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# Check for optional dependencies
HAS_ROUTING_DEPS = False
try:
    import scgraph
    from scgraph.geographs.marnet import marnet_geograph
    HAS_ROUTING_DEPS = True
except ImportError:
    pass


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_maritime_path(origin: Tuple[float, float], dest: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Calculate shortest maritime path using scgraph."""
    if not HAS_ROUTING_DEPS:
        raise ImportError("Dependency 'scgraph' is missing.")

    graph = marnet_geograph
    path_dict = graph.get_shortest_path(
        origin_node={"latitude": origin[0], "longitude": origin[1]},
        destination_node={"latitude": dest[0], "longitude": dest[1]}
    )
    return [(p[0], p[1]) for p in path_dict.get('coordinate_path', [])]


def _interpolate_route(
    path: List[Tuple[float, float]], 
    speed_knots: float, 
    departure: datetime
) -> List[dict]:
    """Interpolate path into 12-hour waypoints."""
    try:
        from geopy.distance import great_circle
    except ImportError:
        # Proper Haversine fallback for accurate distance at all latitudes
        import math
        def great_circle(p1, p2):
            class D:
                def __init__(self, km):
                    self.km = km
            
            lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
            lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            return D(6371 * c)  # Earth radius in km

    speed_kmh = speed_knots * 1.852
    waypoints = []
    current_time = departure
    
    # Add start point
    waypoints.append({
        "lat": path[0][0],
        "lon": path[0][1],
        "time": current_time.strftime("%Y-%m-%d %H:%M"),
        "step": "Origin"
    })

    # We want waypoints roughly every 12 hours for the Agent to check
    target_step_hours = 12
    time_since_last_wp = 0.0
    
    for i in range(1, len(path)):
        prev = path[i-1]
        curr = path[i]
        
        dist = great_circle(prev, curr).km
        hours = dist / speed_kmh if speed_kmh > 0 else 0
        current_time += timedelta(hours=hours)
        time_since_last_wp += hours
        
        # Only add a waypoint if enough time has passed or it's a major turn
        if time_since_last_wp >= target_step_hours:
            waypoints.append({
                "lat": curr[0],
                "lon": curr[1],
                "time": current_time.strftime("%Y-%m-%d %H:%M"),
                "step": "En Route"
            })
            time_since_last_wp = 0
            
    # Always add destination
    if waypoints[-1]["lat"] != path[-1][0]:
        waypoints.append({
            "lat": path[-1][0],
            "lon": path[-1][1],
            "time": current_time.strftime("%Y-%m-%d %H:%M"),
            "step": "Destination"
        })
        
    return waypoints


# ============================================================================
# TOOL FUNCTION
# ============================================================================

def calculate_maritime_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    month: int,
    speed_knots: float = 14.0
) -> str:
    """
    Calculates the detailed maritime route waypoints.
    """
    if not HAS_ROUTING_DEPS:
        return "Error: 'scgraph' not installed."

    try:
        path = _get_maritime_path((origin_lat, origin_lon), (dest_lat, dest_lon))
        
        # Assume travel in the current year or next occurrence of that month
        now = datetime.now()
        year = now.year if month >= now.month else now.year + 1
        departure = datetime(year, month, 15)
        
        waypoints = _interpolate_route(path, speed_knots, departure)
        
        # Format for the Agent
        output = (
            f"MARITIME ROUTE CALCULATED\n"
            f"-------------------------\n"
            f"From: ({origin_lat}, {origin_lon})\n"
            f"To:   ({dest_lat}, {dest_lon})\n"
            f"Total Waypoints: {len(waypoints)}\n\n"
            f"WAYPOINTS (Use these to query ERA5 data):\n"
        )
        
        # Helper bounding box for the Agent
        lats = [w['lat'] for w in waypoints]
        lons = [w['lon'] for w in waypoints]
        output += f"Recommended Bounding Box: Lat [{min(lats)-2:.1f}, {max(lats)+2:.1f}], Lon [{min(lons)-2:.1f}, {max(lons)+2:.1f}]\n\n"
        
        for wp in waypoints:
            output += f"- {wp['time']}: {wp['lat']:.2f}, {wp['lon']:.2f}\n"
            
        output += "\nINSTRUCTION: Now use 'retrieve_era5_data' to check the weather ('swh', 'u10') for these locations."
        
        return output

    except Exception as e:
        return f"Routing Calculation Failed: {str(e)}"


# ============================================================================
# ARGUMENT SCHEMA
# ============================================================================

class RouteArgs(BaseModel):
    origin_lat: float = Field(description="Latitude of origin")
    origin_lon: float = Field(description="Longitude of origin")
    dest_lat: float = Field(description="Latitude of destination")
    dest_lon: float = Field(description="Longitude of destination")
    month: int = Field(description="Month of travel (1-12)")
    speed_knots: float = Field(default=14.0, description="Speed in knots")


# ============================================================================
# LANGCHAIN TOOL
# ============================================================================

routing_tool = StructuredTool.from_function(
    func=calculate_maritime_route,
    name="calculate_maritime_route",
    description="Calculates a realistic maritime route (avoiding land). Returns a list of time-stamped waypoints. DOES NOT check weather.",
    args_schema=RouteArgs
)
