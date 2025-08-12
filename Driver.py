import os
from Essential import DataPreprocessing, FeatureEngineering, SQLDataLoader
from Development import ModelTraining
import warnings
import json
from datetime import datetime, timedelta
import logging

warnings.filterwarnings("ignore")

# ------------------------------------------------- Read the JSON file -------------------------------------------------
with open('config.json', 'r') as file:
    config = json.load(file)

# Access variables
Project_Name = config['Project_Name']
SQL_CSV = config['SQL_CSV']
Feature_Engineering = config['Feature_Engineering']
Normalization = config['Normalization']
custom_threshold = config['custom_threshold']

server = config['server']
database = config['database']
username = config['username']
password = config['password']

Query = config['Query']

Production_Query = config['Production_Query']

# ---------------------------------- Create Model & Reports Folder if they don't exist ---------------------------------
# Define the paths to the directories
script_dir = os.path.dirname(__file__)

# model_dir = os.path.join(script_dir, 'Model')
reports_dir = os.path.join(script_dir, 'Reports')
logs_dir = os.path.join(script_dir, 'Logs')
OUTPUT_dir = os.path.join(script_dir, 'OUTPUT', Project_Name)

# Check if the directories exist, and create them if they don't
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)

if not os.path.exists(reports_dir):
    os.makedirs(reports_dir)

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

if not os.path.exists(OUTPUT_dir):
    os.makedirs(OUTPUT_dir)

# Define the dictionary with the variables
config_data = {
    "Mode": 1,
    "Query": Production_Query
}

filename = OUTPUT_dir + '/'+ Project_Name +'_config.json'

# Write the dictionary to a JSON file
with open(filename, 'w') as json_file:
    json.dump(config_data, json_file, indent=4)


current_date = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

# ------------------------------------------------ Configure the logger -----------------------------------------------
logging.basicConfig(
    # filename=logs_dir + '/log' + current_date + '_' + current_time + '.txt',  # File where logs will be written
    filename=logs_dir + '/' + Project_Name + '_log__' + current_date + '.txt',  # File where logs will be written
    level=logging.INFO,  # Minimum level of log messages to capture
    format='%(asctime)s - %(levelname)s - %(message)s'  # Format of log messages
)

logging.info("------------------------------ ML PIPELINE STARTED ------------------------------")
print('----------------------------------------------------------------------------------------------')
print(f'                                        {Project_Name}                                        ')
print('----------------------------------------------------------------------------------------------')
# ------------------------------------------------ Data Loading Function -----------------------------------------------
logging.info("--------------- ENTERING SQL LOADER MODULE ---------------.")
Data = SQLDataLoader.SQL_Loader(server, database, username, password, Query, current_date, logging)
logging.info("SQL Loader Module Completed Successfully.")
print("Production Data Loaded Successfully")

# ------------------------------------------------- Data Preprocessing -------------------------------------------------
logging.info("--------------- ENTERING DATA PREPROCESSING MODULE ---------------")
PreprocessedData = DataPreprocessing.Preprocessing(Data, logging, SQL_CSV, Project_Name, OUTPUT_dir)
logging.info("Data Preprocessed Module Completed Successfully.")
print("Data Preprocessed Successfully")

# ------------------------------------------------- Feature Engineering ------------------------------------------------
logging.info("--------------- ENTERING FEATURE ENGINEERING MODULE ---------------")
FinalData = FeatureEngineering.FeatureSelection(PreprocessedData, Feature_Engineering, logging, Project_Name, Normalization, OUTPUT_dir)
logging.info("Feature Engineering Module Completed Successfully.")
print("Features Selected Successfully")

# --------------------------------------------------- Model Training ---------------------------------------------------
logging.info("--------------- ENTERING ML TRAINING MODULE ---------------")

try:
    ModelTraining.ModelPreperation(FinalData, custom_threshold, current_date, logging, Project_Name, OUTPUT_dir)
    logging.info("ML Training Module Completed Successfully.")
    print("\n" + "-"*100)
    print(f"{'MODEL TRAINED SUCCESSFULLY':^100}")
    print("-"*100 + "\n")

except Exception as e:
    logging.error(f"Error during model training: {e}", exc_info=True)
    print("MODEL TRAINING FAILED. See log for details.")


logging.info("------------------------------ PIPELINE EXECUTION FINISHED ------------------------------")
