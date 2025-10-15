import pandas as pd
import numpy as np

SHP_FILE_PATH = 'Chi_GIS/Region29.shp' 
SEGMENTS_FILE_PATH = 'Segment_25secW.csv' 
CRASHES_FILE_PATH = 'Traffic_Crashes_-_Crashes_20250117.csv'
FILTERED_CRASHES_OUTPUT_PATH = 'Crashes_25secW.csv'

CRASH_START_DATE = '2025-01-05 00:01:02'
CRASH_END_DATE = '2025-01-11 23:50:25'


try:
    df_crashes_large = pd.read_csv(CRASHES_FILE_PATH, low_memory=False)
    df_crashes_large['timestamp'] = pd.to_datetime(df_crashes_large['CRASH_DATE'], errors='coerce')

    original_rows = len(df_crashes_large)
    df_crashes_large.dropna(subset=['timestamp'], inplace=True)
    if original_rows > len(df_crashes_large):
        print(f"{original_rows - len(df_crashes_large)}")

    mask = (df_crashes_large['timestamp'] >= CRASH_START_DATE) & \
           (df_crashes_large['timestamp'] <= CRASH_END_DATE)
    df_crashes_filtered = df_crashes_large[mask].copy()

    print(f"{len(df_crashes_filtered)}")

except FileNotFoundError:
    print(f"{CRASHES_FILE_PATH}")
except Exception as e:
    print(f"{e}")

try:
    df_segments = pd.read_csv(SEGMENTS_FILE_PATH, low_memory=False)
    print(f" {len(df_segments)}")
    df_segments['timestamp'] = pd.to_datetime(df_segments['TIME'], errors='coerce')
    df_segments.dropna(subset=['timestamp'], inplace=True)
    
    start_time = df_segments['timestamp'].min()
    end_time = df_segments['timestamp'].max()
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='10T')
    print(f"{start_time} to {end_time}")

    all_segments = df_segments['SEGMENT_ID'].unique()
    multi_index = pd.MultiIndex.from_product([all_segments, full_time_range], names=['SEGMENT_ID', 'timestamp'])
    
    df_reindexed = df_segments.set_index(['SEGMENT_ID', 'timestamp']).reindex(multi_index)
    
    df_reindexed['SPEED'].replace(-1, np.nan, inplace=True)
    missing_before = df_reindexed['SPEED'].isna().sum()

    df_reindexed['SPEED_interpolated'] = df_reindexed.groupby('SEGMENT_ID')['SPEED'] \
                                                    .transform(lambda x: x.interpolate(method='linear', limit_direction='both'))
    
    missing_after = df_reindexed['SPEED_interpolated'].isna().sum()

    if missing_after > 0:
        print("The remaining missing values usually refer to those sections where there were no valid speed records throughout the entire period.")

    df_segments_imputed = df_reindexed.reset_index()
    
    df_segments_imputed.dropna(subset=['SPEED_interpolated'], inplace=True)
    
    sample_segment_id = df_segments_imputed['SEGMENT_ID'].iloc[0]
    print(df_segments_imputed[df_segments_imputed['SEGMENT_ID'] == sample_segment_id].head(15))


except FileNotFoundError:
    print(f"{SEGMENTS_FILE_PATH}")
except Exception as e:
    print(f"An error occurred while processing the road section file.: {e}")