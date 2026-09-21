import pandas as pd
import numpy as np
from datetime import datetime

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, sheet_name='Table 1')
    
    date_cols = ['Registration Date', 'Actual Start', 'Actual Finish']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='mixed')
    
    df['duration_hours'] = (df['Actual Finish'] - df['Actual Start']).dt.total_seconds() / 3600.0
    
    # Handle zero-duration rows: replace 0 with 0.25 hours (15 min minimum)
    df.loc[df['duration_hours'] == 0, 'duration_hours'] = 0.25
    
    def get_cat(text):
        if pd.isna(text): return 'Other'
        t = str(text).lower()
        if any(w in t for w in ['torch', 'plasma', 'consumable', 'ignition']) or ('coolant' in t and not any(w in t for w in ['dust', 'vacuum'])):
            return 'Plasma/Torch'
        if any(w in t for w in ['conveyor', 'roller', 'infeed', 'outfeed', 'chain', 'lifter', 'transfer']):
            return 'Conveyor/Roller'
        if any(w in t for w in ['laser', 'sensor', 'scanning', 'measurement', 'scan']):
            return 'Laser/Sensor'
        if any(w in t for w in ['dust', 'vacuum', 'filter', 'diaphragm', 'smoke']):
            return 'Dust Collector'
        if 'hydraulic' in t or 'hose' in t:
            return 'Hydraulic'
        if 'oil' in t:
            return 'Hydraulic'
        if any(w in t for w in ['alarm', 'program', 'internet', 'panel', 'amplifier', 'fuse', 'robot', 'calibration', 'motion', 'axis']):
            return 'Control/Electrical'
        if any(w in t for w in ['cover', 'telescopic', 'clamp', 'rack', 'pinion', 'bearing', 'bellow']):
            return 'Mechanical'
        return 'Other'
        
    df['category'] = df['Directive'].apply(get_cat)
    
    rename_map = {
        'WO No': 'wo_no',
        'WO Site': 'wo_site',
        'Registration Date': 'registration_date',
        'Object ID': 'object_id',
        'Object Description': 'object_description',
        'Directive': 'directive',
        'Actual Start': 'actual_start',
        'Actual Finish': 'actual_finish',
        'Work Typ': 'work_type_code',
        'Work Typ.1': 'work_type',
        'Work Details': 'work_details'
    }
    df = df.rename(columns=rename_map)
    
    df = df.sort_values('actual_start', ascending=True).reset_index(drop=True)
    return df

def get_data_quality_report(df: pd.DataFrame) -> dict:
    return {
        'total_records': len(df),
        'date_range_start': df['actual_start'].min(),
        'date_range_end': df['actual_start'].max(),
        'null_counts': df.isnull().sum().to_dict(),
        'zero_duration_count': (df['duration_hours'] == 0.25).sum(),
        'extreme_duration_count': (df['duration_hours'] > 200).sum(),
        'work_type_distribution': df['work_type'].value_counts().to_dict(),
        'category_distribution': df['category'].value_counts().to_dict(),
        'registration_vs_start_mismatch_count': (df['registration_date'] != df['actual_start']).sum()
    }
