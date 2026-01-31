
import os
import sys
import time
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from pathlib import Path

# Add src to path to import vostok tools
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from vostok.tools.era5 import retrieve_era5_data
from vostok.config import DATA_DIR, PLOTS_DIR

def run_oceanography_workflow():
    print("===========================================================")
    print("🧪 STARTING OCEANOGRAPHY WORKFLOW TEST")
    print("===========================================================")

    # 1. SETUP
    # ------------------------------------------------------------------
    region = "gulf_of_mexico"
    start_date = "2023-01-01"
    end_date = "2023-01-07"
    variable = "sst"
    
    print(f"\n[1] Retrieving {variable} data for {region} ({start_date} to {end_date})...")
    
    # Check if we should force download or rely on tool caching
    # We'll just call the tool, it handles caching.
    result = retrieve_era5_data(
        query_type="temporal",
        variable_id=variable,
        start_date=start_date,
        end_date=end_date,
        region=region
    )
    
    print(f"Tool Result:\n{result}")
    
    if "Error" in result:
        print("❌ Data retrieval failed!")
        sys.exit(1)
        
    # Extract path from result (it's in the text output of the tool)
    # The tool returns text like "Path: data/era5_..."
    try:
        path_line = [line for line in result.split('\n') if "Path:" in line][0]
        zarr_path = path_line.split("Path:")[1].strip()
    except IndexError:
        print("❌ Could not parse Zarr path from tool output.")
        sys.exit(1)
        
    print(f"✅ Data located at: {zarr_path}")

    # 2. ANALYSIS
    # ------------------------------------------------------------------
    print("\n[2] Performing Analysis...")
    
    try:
        ds = xr.open_dataset(zarr_path, engine='zarr')
        
        # Basic stats
        sst = ds[variable]
        mean_val = sst.mean().item()
        max_val = sst.max().item()
        min_val = sst.min().item()
        
        print(f"   Shape: {sst.shape}")
        print(f"   Mean SST: {mean_val:.2f} K")
        print(f"   Max SST:  {max_val:.2f} K")
        print(f"   Min SST:  {min_val:.2f} K")
        
        # Check for anomalies (simple threshold)
        high_temp_mask = sst > 300 # > 27C approx
        high_temp_count = high_temp_mask.sum().item()
        print(f"   Grid points > 300K: {high_temp_count}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        sys.exit(1)

    # 3. VISUALIZATION (MAP)
    # ------------------------------------------------------------------
    print("\n[3] Generating Map...")
    plots_dir = Path("data/plots_test")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Plot the first time step
        plt.figure(figsize=(10, 6))
        sst.isel(time=0).plot(cmap='RdYlBu_r')
        plt.title(f"SST - Gulf of Mexico - {start_date}")
        map_path = plots_dir / "sst_map_step0.png"
        plt.savefig(map_path)
        plt.close()
        print(f"✅ Map saved to: {map_path}")
        
    except Exception as e:
        print(f"❌ Map generation failed: {e}")
        sys.exit(1)

    # 4. VIDEO GENERATION (FFMPEG)
    # ------------------------------------------------------------------
    print("\n[4] Generating Video Frames...")
    
    frames_dir = plots_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate a frame for each time step
        # Limit to 24 frames max to save time if dataset is huge
        steps = min(len(sst.time), 24)
        
        for i in range(steps):
            plt.figure(figsize=(10, 6))
            sst.isel(time=i).plot(cmap='RdYlBu_r', add_colorbar=True)
            
            # Timestamp title
            ts = pd.to_datetime(sst.time[i].values)
            plt.title(f"SST - {ts.strftime('%Y-%m-%d %H:%M')}")
            
            frame_name = f"frame_{i:03d}.png"
            plt.savefig(frames_dir / frame_name)
            plt.close()
            
            if i % 5 == 0:
                print(f"   Generated frame {i}/{steps}")
                
        print(f"✅ {steps} frames generated in {frames_dir}")
        
        # Run FFmpeg
        print("\n[5] Encoding Video with FFmpeg...")
        video_path = plots_dir / "sst_animation.mp4"
        
        # Check if ffmpeg is installed
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            cmd = [
                "ffmpeg",
                "-y", # Overwrite
                "-framerate", "5",
                "-i", str(frames_dir / "frame_%03d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path)
            ]
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"✅ Video saved to: {video_path}")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("⚠️ FFmpeg not found or failed. Skipping video encoding.")
            print("   Frames are available for manual inspection.")

    except Exception as e:
        print(f"❌ Video generation logic failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n===========================================================")
    print("🎉 WORKFLOW TEST COMPLETE")
    print("===========================================================")

if __name__ == "__main__":
    # Ensure pandas is imported for timestamp handling in loop
    import pandas as pd
    run_oceanography_workflow()
