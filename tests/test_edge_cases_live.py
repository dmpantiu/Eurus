
import os
import sys
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from vostok.tools.era5 import retrieve_era5_data

def test_future_date():
    print("Testing Future Date Retrieval (Live API)...")
    
    # Date 30 days in the future
    future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    
    result = retrieve_era5_data(
        query_type="temporal",
        variable_id="t2",
        start_date=future_date,
        end_date=future_date,
        region="gulf_of_mexico"
    )
    
    print("Result:")
    print(result)
    
    if "Error" in result:
        print("✅ Correctly returned error for future date.")
    else:
        print("⚠️ Unexpected success or silence for future date.")

if __name__ == "__main__":
    test_future_date()
