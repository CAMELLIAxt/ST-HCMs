import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree
import numpy as np

SHP_FILE_PATH = 'Chi_GIS/Region29.shp'
SEGMENTS_FILE_PATH = 'Segment_25secW.csv' 
FILTERED_CRASHES_PATH = 'Crashes_25secW.csv'
FINAL_PANEL_OUTPUT_PATH = 'chicago_panel_final.csv'

DATETIME_FORMAT = '%m/%d/%Y %I:%M:%S %p'


def load_and_impute_segments(file_path):
    df_segments = pd.read_csv(file_path, low_memory=False)
    df_segments['timestamp_raw'] = pd.to_datetime(df_segments['TIME'], format=DATETIME_FORMAT, errors='coerce')
    df_segments.dropna(subset=['timestamp_raw'], inplace=True)
    df_segments['timestamp'] = df_segments['timestamp_raw'].dt.floor('10min')
    df_segments = df_segments.groupby(['SEGMENT_ID', 'timestamp'])['SPEED'].mean().reset_index()
    
    start_time, end_time = df_segments['timestamp'].min(), df_segments['timestamp'].max()
    full_time_range = pd.date_range(start=start_time, end=end_time, freq='10min')
    
    all_segments_ids = df_segments['SEGMENT_ID'].unique()
    multi_index = pd.MultiIndex.from_product([all_segments_ids, full_time_range], names=['SEGMENT_ID', 'timestamp'])
    
    df_reindexed = df_segments.set_index(['SEGMENT_ID', 'timestamp']).reindex(multi_index)
    df_reindexed['SPEED'] = df_reindexed['SPEED'].replace(-1, np.nan)
    
    df_reindexed['speed'] = df_reindexed.groupby('SEGMENT_ID')['SPEED'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    
    df_imputed = df_reindexed.reset_index()
    df_imputed.dropna(subset=['speed'], inplace=True)
    
    df_segment_coords = pd.read_csv(file_path, usecols=['SEGMENT_ID', 'START_LONGITUDE', 'START_LATITUDE', 'END_LONGITUDE', 'END_LATITUDE']).drop_duplicates('SEGMENT_ID')
    return df_imputed, df_segment_coords

def assign_crashes_to_segments(crashes_path, regions_gdf, segments_map, segments_coords):
    df_crashes = pd.read_csv(crashes_path, low_memory=False)
    df_crashes['timestamp'] = pd.to_datetime(df_crashes['CRASH_DATE'], format=DATETIME_FORMAT, errors='coerce')
    df_crashes.dropna(subset=['timestamp', 'LONGITUDE', 'LATITUDE'], inplace=True)
    
    df_crashes['timestamp_10min'] = df_crashes['timestamp'].dt.floor('10min')

    gdf_crashes = gpd.GeoDataFrame(
        df_crashes, 
        geometry=gpd.points_from_xy(df_crashes.LONGITUDE, df_crashes.LATITUDE),
        crs="EPSG:4283"
    )
    
    crashes_in_regions = gpd.sjoin(gdf_crashes, regions_gdf, how="inner", predicate="within")

    unique_crashes_in_regions = crashes_in_regions['CRASH_RECORD_ID'].nunique()
    print(f"Debug: {unique_crashes_in_regions} / {df_crashes['CRASH_RECORD_ID'].nunique()} ")
    if len(crashes_in_regions) == 0:
        return pd.DataFrame(columns=['timestamp', 'SEGMENT_ID', 'treatment'])

    assigned_crashes = []
    unique_segments = segments_coords.drop_duplicates('SEGMENT_ID').copy()
    unique_segments['midpoint_lon'] = (unique_segments['START_LONGITUDE'] + unique_segments['END_LONGITUDE']) / 2
    unique_segments['midpoint_lat'] = (unique_segments['START_LATITUDE'] + unique_segments['END_LATITUDE']) / 2

    for unit_id, group in crashes_in_regions.groupby('unit_id'):
        segments_in_unit = segments_map[segments_map['unit_id'] == unit_id]['SEGMENT_ID']
        unit_segments_geo = unique_segments[unique_segments['SEGMENT_ID'].isin(segments_in_unit)]
        
        if len(unit_segments_geo) == 0: continue

        tree = cKDTree(np.radians(unit_segments_geo[['midpoint_lat', 'midpoint_lon']].values))
        crash_coords = np.radians(group[['LATITUDE', 'LONGITUDE']].values)
        _, indices = tree.query(crash_coords, k=1)
        
        group['assigned_segment_id'] = unit_segments_geo['SEGMENT_ID'].iloc[indices].values
        assigned_crashes.append(group[['timestamp_10min', 'assigned_segment_id']])

    if not assigned_crashes:
        return pd.DataFrame(columns=['timestamp', 'SEGMENT_ID', 'treatment'])

    df_crashes_final = pd.concat(assigned_crashes)
    df_crashes_agg = df_crashes_final.groupby(['timestamp_10min', 'assigned_segment_id']).size().reset_index(name='crash_count')
    df_crashes_agg['treatment'] = 1
    
    return df_crashes_agg[['timestamp_10min', 'assigned_segment_id', 'treatment']]

gdf_regions = gpd.read_file(SHP_FILE_PATH).to_crs("EPSG:4283")
if 'REGION_ID' in gdf_regions.columns:
    gdf_regions.rename(columns={'REGION_ID': 'unit_id'}, inplace=True)

df_segments_imputed, df_segment_coords = load_and_impute_segments(SEGMENTS_FILE_PATH)

gdf_starts = gpd.GeoDataFrame(df_segment_coords, geometry=gpd.points_from_xy(df_segment_coords.START_LONGITUDE, df_segment_coords.START_LATITUDE), crs="EPSG:4283")
gdf_ends = gpd.GeoDataFrame(df_segment_coords, geometry=gpd.points_from_xy(df_segment_coords.END_LONGITUDE, df_segment_coords.END_LATITUDE), crs="EPSG:4283")
segment_to_unit_map = pd.concat([
    gpd.sjoin(gdf_starts, gdf_regions[['unit_id', 'geometry']], how="inner", predicate="within")[['SEGMENT_ID', 'unit_id']],
    gpd.sjoin(gdf_ends, gdf_regions[['unit_id', 'geometry']], how="inner", predicate="within")[['SEGMENT_ID', 'unit_id']]
]).drop_duplicates()

df_treatment = assign_crashes_to_segments(FILTERED_CRASHES_PATH, gdf_regions, segment_to_unit_map, df_segment_coords)
df_treatment.rename(columns={'timestamp_10min': 'timestamp', 'assigned_segment_id': 'SEGMENT_ID'}, inplace=True)

df_panel = df_segments_imputed[['timestamp', 'SEGMENT_ID', 'speed']]
df_panel = pd.merge(df_panel, segment_to_unit_map, on='SEGMENT_ID', how='inner')

df_panel['SEGMENT_ID'] = df_panel['SEGMENT_ID'].astype(int)
df_treatment['SEGMENT_ID'] = df_treatment['SEGMENT_ID'].astype(int)

df_panel = pd.merge(df_panel, df_treatment, on=['timestamp', 'SEGMENT_ID'], how='left')
df_panel['treatment'] = df_panel['treatment'].fillna(0)
df_panel['treatment'] = df_panel['treatment'].astype(int)

df_panel.rename(columns={'unit_id': 'unit', 'SEGMENT_ID': 'subunit'}, inplace=True)
final_panel = df_panel[['timestamp', 'unit', 'subunit', 'speed', 'treatment']]
final_panel = final_panel.sort_values(by=['unit', 'subunit', 'timestamp']).reset_index(drop=True)