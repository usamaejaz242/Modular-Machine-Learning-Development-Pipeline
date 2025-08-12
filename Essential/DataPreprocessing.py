import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import joblib
import os

def Preprocessing(Data, logging, SQL_CSV, Project_Name, OUTPUT_dir):
    if 'RPRTNG_DT' in Data.columns:
        Data = Data.drop(['RPRTNG_DT'], axis=1)
    # else:
    #     Data = Data.drop(Data.columns[[0]], axis=1)
    # if 'MSISDN' in Data.columns:
    #     Data = Data.drop(['MSISDN'], axis=1)
    # else:
    #     Data = Data.drop(Data.columns[[2]], axis=1)
    # if SQL_CSV == 1:
    #     if 'CHURN_FLAG' in Data.columns:
    #         Data = Data.drop(['CHURN_FLAG'], axis=1)
    # else:
    #     Data = Data.drop(Data.columns[[0, 2, -1]], axis=1)
    logging.info("Date and MSISDN removed.")

    # Remove rows with missing values
    Data = Data.dropna()
    logging.info("Null values dropped.")

    # Read the Customer_ID column
    Customer_ID = Data['MSISDN']

    # Separate the target column and Customer_ID
    # y = Data[Customer_ID]
    df_X = Data.drop(['MSISDN'], axis=1)

# --------------------------------------- IF Boolean Columns including Target Class ------------------------------------------------
    # Map boolean columns to 0 and 1
    logging.info("Checking if features are of boolean datatype.")
    bool_check = 0
    for col in df_X.columns:
        if df_X[col].dtype == 'bool':
            bool_check += 1
            df_X[col] = Data[col].astype(int)

    if bool_check >= 1:
        logging.info("YES, features are of boolean datatype.")
        logging.info("Converting boolean features to integers.")
        logging.info("Conversion Completed.")
    else:
        logging.info("NO, features are not of boolean datatype.")
    # --------------------------------------- IF Target Class Encoding is Required ---------------------------------------
    # Check if the last column is of type object (string)
    # if df_X.dtypes[-1] == 'object' or df_X.dtypes[-1] == 'bool':
    logging.info("Checking if target class needs class encoding.")
    # if df_X.dtypes[-1] == 'object':
    if df_X.dtypes['FLAG'] == 'object':
        logging.info("YES, target class needs class encoding.")

        # Perform label encoding
        logging.info("Initializing Label Encoder.")
        label_encoder = LabelEncoder()
        df_X.iloc[:, -1] = label_encoder.fit_transform(df_X.iloc[:, -1])
        # label_encoder.fit_transform(X[string_columns])
        # encoded_data = onehot_encoder.transform(X[string_columns])

        logging.info("Converting target class values to integer values using Label Encoder.")
        target_column = 'FLAG'
        df_X[target_column] = df_X[target_column].astype(str).astype(int)
        logging.info("Conversion Completed.")
    else:
        logging.info("NO, target class does not needs class encoding.")

        # print(df_X['Dormant'].dtype)
        # print(df_X.dtypes[-1])

# --------------------------------------- IF NO STRING Columns ------------------------------------------------
    # Select columns with string values (labels)
    logging.info("Checking if One-Hot-Encoding is required.")
    string_columns = df_X.select_dtypes(include=['object', 'bool']).columns

    if len(string_columns) == 0:
        logging.info("NO, One-Hot-Encoding is not required.")
        df_X = pd.concat([Data['MSISDN'], df_X], axis=1)
        return df_X


# --------------------------------------- IF STRING Columns (One-Hot Encoding) --------------------------------
    else:
        logging.info("YES, One-Hot-Encoding is required.")

        # Separate the target column and Customer_ID
        X = df_X.drop(['FLAG'], axis=1)

        logging.info("Initializing One-Hot-Encoder Model.")

        # One-hot encoding for string columns
        onehot_encoder = OneHotEncoder(sparse_output=False)
        logging.info("Transforming Categorical Features to Numerical Features.")
        onehot_encoder.fit_transform(X[string_columns])
        encoded_data = onehot_encoder.transform(X[string_columns])
        logging.info("Saving the One-Hot-Encoder Model.")
        # script_dir = os.path.dirname(__file__)
        encoder_path = os.path.join(OUTPUT_dir, Project_Name + '_one_hot_encoder.pkl')
        joblib.dump(onehot_encoder, encoder_path)
        logging.info("Saved.")

        # Create a dataframe with the encoded data
        encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoder.get_feature_names_out(string_columns))

        # Reset the index of the encoded dataframe to align with the original dataframe
        encoded_df.index = X.index

        # Drop the original string columns
        df_X2 = X.drop(string_columns, axis=1)

        # Concatenate the encoded dataframe with the original dataframe
        df_X2 = pd.concat([df_X2, encoded_df], axis=1)
        df_X2 = pd.concat([Data['MSISDN'], df_X2], axis=1)
        df_X2 = pd.concat([df_X2, df_X['FLAG']], axis=1)
        # df_X[target_column] = y
        df_X2.reset_index(drop=True)

        logging.info("Encoding Completed.")

        return df_X2.reset_index(drop=True)
