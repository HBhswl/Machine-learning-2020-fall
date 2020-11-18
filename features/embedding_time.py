import pandas as pd
import numpy as pd

def embedding_time(df):
    df['Dates-year'] = df['Dates'].dt.year
    df['Dates-month'] = df['Dates'].dt.month
    df['Dates-day'] = df['Dates'].dt.day
    df['Dates-hour'] = df['Dates'].dt.hour
    
    df['DayOfWeek'] = df['Dates'].dt.dayofweek
    df['WeekOfYear'] = df['Dates'].dt.weekofyear #isocalendar().week.astype(np.int64)
    
    df.drop(columns=["Dates"], inplace=True)
    return df

