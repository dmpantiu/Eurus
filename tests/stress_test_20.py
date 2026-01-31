import os
import sys
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from vostok.tools.era5 import retrieve_era5_data

OUTPUT_FILE = "test_reports/stress_test_results.txt"

QUERIES = [
    # 1-5
    {"variable_id": "sst", "region": "gulf_of_mexico", "start_date": "2023-01-01", "end_date": "2023-01-01", "query_type": "spatial"},
    {"variable_id": "t2", "region": "california_coast", "start_date": "2023-01-01", "end_date": "2023-01-01", "query_type": "spatial"},
    {"variable_id": "u10", "region": "caribbean", "start_date": "2023-01-01", "end_date": "2023-01-01", "query_type": "spatial"},
    {"variable_id": "v10", "region": "mediterranean", "start_date": "2023-01-01", "end_date": "2023-01-01", "query_type": "spatial"},
    {"variable_id": "mslp", "region": "nino34", "start_date": "2023-01-01", "end_date": "2023-01-01", "query_type": "spatial"},
    
    # 6-10 (Different variables/regions)
    {"variable_id": "tcc", "region": "nino3", "start_date": "2023-02-01", "end_date": "2023-02-01", "query_type": "spatial"},
    {"variable_id": "tp", "region": "nino4", "start_date": "2023-02-01", "end_date": "2023-02-01", "query_type": "spatial"},
    {"variable_id": "sp", "region": "east_coast_us", "start_date": "2023-02-01", "end_date": "2023-02-01", "query_type": "spatial"},
    {"variable_id": "sst", "region": "australia", "start_date": "2023-02-01", "end_date": "2023-02-01", "query_type": "spatial"},
    {"variable_id": "t2", "region": "asia_east", "start_date": "2023-02-01", "end_date": "2023-02-01", "query_type": "spatial"},

    # 11-15 (Large regions, but only 1 day to keep it manageable)
    {"variable_id": "u10", "region": "north_atlantic", "start_date": "2023-03-01", "end_date": "2023-03-01", "query_type": "spatial"},
    {"variable_id": "v10", "region": "south_atlantic", "start_date": "2023-03-01", "end_date": "2023-03-01", "query_type": "spatial"},
    {"variable_id": "mslp", "region": "north_pacific", "start_date": "2023-03-01", "end_date": "2023-03-01", "query_type": "spatial"},
    {"variable_id": "tcc", "region": "south_pacific", "start_date": "2023-03-01", "end_date": "2023-03-01", "query_type": "spatial"},
    {"variable_id": "tp", "region": "indian_ocean", "start_date": "2023-03-01", "end_date": "2023-03-01", "query_type": "spatial"},

    # 16-20 (Temporal queries for small regions)
    {"variable_id": "sst", "region": "nino12", "start_date": "2023-04-01", "end_date": "2023-04-02", "query_type": "temporal"},
    {"variable_id": "t2", "region": "gulf_of_mexico", "start_date": "2023-04-01", "end_date": "2023-04-02", "query_type": "temporal"},
    {"variable_id": "u10", "region": "california_coast", "start_date": "2023-04-01", "end_date": "2023-04-02", "query_type": "temporal"},
    {"variable_id": "v10", "region": "caribbean", "start_date": "2023-04-01", "end_date": "2023-04-02", "query_type": "temporal"},
    {"variable_id": "mslp", "region": "europe", "start_date": "2023-04-01", "end_date": "2023-04-01", "query_type": "spatial"} # Back to spatial for Europe
]

def run_stress_test():
    print(f"Starting Stress Test: {len(QUERIES)} requests...")
    
    # Ensure output dir exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"STRESS TEST REPORT - {datetime.now().isoformat()}\n")
        f.write("="*60 + "\n\n")

    for i, q in enumerate(QUERIES, 1):
        print(f"Processing Request {i}/{len(QUERIES)}: {q['variable_id']} - {q['region']}...")
        
        start_time = time.time()
        try:
            result = retrieve_era5_data(**q)
        except Exception as e:
            result = f"CRITICAL EXCEPTION: {e}"
        duration = time.time() - start_time
        
        # Log to file
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"REQUEST #{i}\n")
            f.write(f"Query: {q}\n")
            f.write(f"Duration: {duration:.2f}s\n")
            f.write(f"Result:\n{result}\n")
            f.write("-" * 60 + "\n\n")
            
        # Optional: brief pause to be nice to the API?
        # Earthmover is robust, but let's do 0.5s
        time.sleep(0.5)

    print(f"\n✅ Stress test complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_stress_test()
