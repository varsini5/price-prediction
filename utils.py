import pandas as pd

def preprocess_input(data):
    """
    Preprocess input data by converting date columns to datetime 
    and creating new numerical features for machine learning.
    """
    # Ensure data is in DataFrame format
    df = pd.DataFrame([data])  # Convert dict to DataFrame for processing

    # Define the date columns
    date_columns = ['ETD', 'ATD', 'ETA', 'ATA']

    # Convert date columns to datetime format
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%d-%m-%Y", errors='coerce')

    # Create new numerical features
    df['Transit_Time'] = (df['ETA'] - df['ETD']).dt.days
    df['Delay_ATD'] = (df['ATD'] - df['ETD']).dt.days
    df['Delay_ATA'] = (df['ATA'] - df['ETA']).dt.days

    # Drop original date columns
    df.drop(columns=date_columns, inplace=True)

    return df.iloc[0].to_dict()  # Return processed data as a dictionary
