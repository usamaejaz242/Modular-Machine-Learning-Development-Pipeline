import pyodbc
import pandas as pd
import time
import warnings
import os
import json
with open("config.json", "r") as f:
    config = json.load(f)

warnings.filterwarnings("ignore")
SQL_CSV = config.get("SQL_CSV", 0)
def SQL_Loader(server, database, username, password,Query, current_date, logging):

    if SQL_CSV == 1:
        csv_path = config.get("csv_path", "PLAN_DORMANCY_ADS_data.csv") 
        df = pd.read_csv(csv_path)
        logging.info(f"Data Reading Completed from CSV: {csv_path}")
        return df.reset_index(drop=True)

        # df['FLAG'] = (df['TOTL_OG_SMS_SUCCESS'] > 2).astype(int)
        # logging.info("Label Assignment Completed.")

        return df.reset_index(drop=True)

    if SQL_CSV == 2:
        data = {
            "ACCOUNT": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Age": [25, 30, 35, 20, 28, 40, 32, 38, 45, 29],
            "Balance": [1000, 5000, 2000, 3000, 10000, 500, 8000, 2000, 1500, 4000],
            "CreditScore": [700, 600, 750, 650, 800, 550, 700, 600, 750, 650],
            "IsActive": ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],
            "NumOfTransactions": [5, 10, 2, 8, 15, 3, 12, 4, 9, 6],
            "FLAG": ['No', 'No', 'Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'No']
        }

        df = pd.DataFrame(data)
        logging.info("Data Reading Completed.")
        return df.reset_index(drop=True)

    # Establish a connection to the SQL Server
    conn = pyodbc.connect(f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')

    # Measure the start time
    start_time = time.time()

    # Read the data into a pandas DataFrame
    df = pd.read_sql(Query, conn)
    logging.info("Connection Built and Data Reading Completed.")

    # df['Churn'] = (df['TOTL_OG_SMS_SUCCESS'] > 2).astype(int)
    # logging.info("Label Assignment Completed.")

    # Measure the end time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Close the connection
    conn.close()
    logging.info("Connection Closure Completed.")


    print(f"Query executed and data loaded in {elapsed_time:.2f} seconds")
    return df.reset_index(drop=True)

