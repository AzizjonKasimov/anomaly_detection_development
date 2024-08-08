
import pandas as pd

class Data_Formatter():
    def __init__(self, df):
        self.df = self.change_column_names(df)
        self.df = self.resample(self.df) 

    def change_column_names(self, df):
        rename_dict = {
                'SEQ':  'sequence',
                'UNIX_TIMESTAMP':  'unix_timestamp',
                'REG_DATETIME':  'regular_time',
                'TIMESTAMP':  'timestamp',
                'A_R': 'r_phase_current',
                'A_S': 's_phase_current',
                'A_T': 't_phase_current',
                'A_PV': 'pv_current',
                'HZ': 'system_frequency',
                'KW': 'generated_power',
                'KWH': 'total_generated_power',
                'SOLAR_H': 'horizontal_solar_radiation',
                'SOLAR_V': 'vertical_solar_radiation',
                'T_M': 'module_temperature',
                'T_O': 'outdoor_temperature',
                'V_PV': 'pv_voltage',
                'V_LL_RS': 'rs_line_voltage',
                'V_LL_ST': 'st_line_voltage',
                'V_LL_TR': 'tr_line_voltage',
                'W_PV': "pv_generated_power",
                'OP_STATUS': 'operation_status',
                'V_GRID_INFO': 'rated_grid_voltage'
            }
        
        #check if at least one column name matches the column list
        for column_name in rename_dict.keys():
            if column_name in df.columns.to_list():
                df.rename(columns={column_name: rename_dict[column_name]}, inplace=True)
        return df
    
    def resample(self, df):
        df.index = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
        df = df.resample('10min').agg({
            'r_phase_current': 'mean',
            's_phase_current': 'mean',
            't_phase_current': 'mean',
            'pv_current': 'mean',
            'pv_voltage': 'mean',
            'generated_power': 'sum',
            'pv_generated_power': 'sum',
            'horizontal_solar_radiation': 'sum',
            'vertical_solar_radiation': 'sum',
            'module_temperature': 'mean',
            'outdoor_temperature': 'mean',
            'rs_line_voltage': 'mean',
            'st_line_voltage': 'mean',
            'tr_line_voltage': 'mean'
            })
        df = df.reset_index()

        # Drop rows with NaN values
        df = df.dropna().drop_duplicates()

        return df



# ------------------------------------------------------------------------------------------------------------------------------------------------------

class Data_Preprocessor():
    def __init__(self, df, scaler, train_data=False):
        self.train_data = train_data
        self.df = self.clean_data(df)
        self.df = self.fix_temperature_values(self.df)
        self.df = self.cap_outliers(self.df, self.df.columns, lower_percentile=0.01, upper_percentile=0.99)
        self.df = self.normalize_data(self.df, scaler)

    def fix_temperature_values(self, df):
        #Apply the replacement of the outliers for temperature values
        columns = ['module_temperature', 'outdoor_temperature']
        for column in columns:
            df[column] = df[column].apply(lambda x: (300 + x) * -1 if x < -50 else x)
        return df


    def cap_outliers(self, df, columns, lower_percentile=0.0001, upper_percentile=0.9999):
        """
        Caps the outliers in specified columns of a DataFrame based on the specified lower and upper percentiles.
        
        Parameters:
        - df: pandas.DataFrame containing the data.
        - columns: list of str, the names of the columns to cap.
        - lower_percentile: float, the lower percentile to use for capping, expressed as a decimal between 0 and 1.
        - upper_percentile: float, the upper percentile to use for capping, expressed as a decimal between 0 and 1.
        
        Returns:
        - A pandas.DataFrame with the outliers in the specified columns capped.
        """
        # Copy the DataFrame to avoid modifying the original data
        new_df = df.copy()
        
        for column in columns:
            if column == "timestamp": # Skip the timestamp column
                continue
            # Determine the percentile values for capping
            lower_bound = new_df[column].quantile(lower_percentile)
            upper_bound = new_df[column].quantile(upper_percentile)
            
            # Cap values outside the interpercentile range
            new_df[column] = new_df[column].clip(lower_bound, upper_bound)
        
        return new_df
    
    def clean_data(self, df):
        df = df.dropna() # drop the NaN values

        df_copy = df.copy() # use a copy of the DataFrame to avoid SettingWithCopyWarning 
        for col in df.drop('timestamp', axis=1).select_dtypes(include=['int', 'int64']).columns:  # exclude the timestamp column and convert the rest to float
            df.loc[:, col] = df_copy[col].astype(float)  

        # convert timestamps to datetime if the file has timestamp values in "YYYYMMDDHHMMSS" format
        df.index = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M%S')
        df = df.drop('timestamp', axis=1)

        # get rid of duplicate indexes
        df = df[~df.index.duplicated(keep='first')] # here ~ inverts the Booleans returned by this condition
        
        return df
    
    def normalize_data(self, df, scaler):
        if self.train_data:
            df_transformed = scaler.fit_transform(df)
            df = pd.DataFrame(df_transformed, columns=df.columns, index=df.index)
        else:
            df_transformed = scaler.transform(df)
            df = pd.DataFrame(df_transformed, columns=df.columns, index=df.index)
        return df
    

def save_data(df, file_path):
    df.to_csv(file_path, index=False)