from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import joblib

import statsmodels.api as sm
import pandas as pd


def ensure_columns(df, columns_from_file):
    # Identify missing columns
    missing_columns = [col for col in columns_from_file if col not in df.columns]

    # Create a DataFrame with missing columns filled with zeros
    if missing_columns:
        missing_df = pd.DataFrame(0, index=df.index, columns=missing_columns)
        df = pd.concat([df, missing_df], axis=1)

    # Reorder the columns to match the order in columns_from_file
    df = df[columns_from_file]

    return df


def BackwardElimination_Features(X, y, significance_level=0.05):
    # Adding a constant column to the dataset for the intercept
    X = sm.add_constant(X)

    # Creating the initial model with all features
    model = sm.OLS(y, X).fit()

    # Loop to remove features iteratively based on p-values
    while True:
        # Getting the p-values of all features
        p_values = model.pvalues

        # Finding the feature with the highest p-value
        max_p_value = p_values.max()

        # If the highest p-value is greater than the significance level, remove the feature
        if max_p_value > significance_level:
            feature_to_remove = p_values.idxmax()
            X = X.drop(columns=[feature_to_remove])

            # Re-fitting the model with the remaining features
            model = sm.OLS(y, X).fit()
        else:
            break

    if 'const' in X.columns:
        # Returning the selected features (excluding the constant column)
        selected_features = X.columns.drop('const')
    else:
        selected_features = X
        selected_features = selected_features.columns

    return selected_features, model.summary()


# Example usage:
# selected_features, summary = BackwardElimination(X_train, y_train)
# print("Selected Features:", selected_features)
# print(summary)


def RFECV_Features(X, y):
    # Initialize a RandomForestClassifier for RFECV
    model = RandomForestClassifier(random_state=24)

    # Initialize RFECV with the model
    rfecv = RFECV(estimator=model, step=1, cv=3,
                  scoring='accuracy')  # You can adjust the cross-validation and scoring method

    # Fit RFECV
    rfecv.fit(X, y)

    # Get the selected features
    selected_features = X.columns[rfecv.support_]

    return selected_features


def FeatureSelection(df, Feature_Engineering, logging, Project_Name, Normalization, OUTPUT_dir):
    # --------------------------------------- For Model Training ---------------------------------------
    global X_Normalized
    df.reset_index(drop=True)

    # Separate the target column and Customer_ID
    X = df.drop(['MSISDN', 'FLAG'], axis=1)
    # X = df_X.drop(target_column, axis=1)
    y = df['FLAG']

    # --------------------------------------------- Data Normalization -------------------------------------------------
    if Normalization == 1:
        logging.info("Normalizing the data.")
        scaler = StandardScaler()
        # Step 2: Fit the scaler to the DataFrame and transform the data
        # It's common to exclude non-numeric columns from scaling
        df_numeric = X.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
        scaler_model = scaler.fit(df_numeric)
        scaled_data = scaler.fit_transform(df_numeric)
        # Step 3: Convert the scaled data back to a DataFrame
        df_scaled = pd.DataFrame(scaled_data, columns=df_numeric.columns)
        # If you want to include non-numeric columns back into the DataFrame
        X_Normalized = df_scaled.join(X[X.select_dtypes(exclude=['float64', 'int64']).columns])
        logging.info("Normalized.")

        logging.info("Saving the Standard Scaler Model.")
        # script_dir = os.path.dirname(__file__)
        scaler_path = os.path.join(OUTPUT_dir, Project_Name + '_scaler.pkl')
        joblib.dump(scaler_model, scaler_path)
        logging.info("Saved.")
    else:
        X_Normalized = X
    # --------------------------------------------------------------------------------------------------------------

    logging.info("Checking if Feature Engineering is ON or OFF.")
    if Feature_Engineering == 1:
        X_Normalized.index = y.index
        logging.info("YES, Feature Engineering is ON.")
        logging.info("Performing Feature Engineering.")
        #  Feature selection using Backward Selection
        # selected_features, Model_Summary = BackwardElimination_Features(X_Normalized, y)
        # print('Model_Summary:', Model_Summary)
        # selected_features = RFECV_Features(X_Normalized, y)

        selected_features = ['M1_PKG_OG_VCE_CNT',
                             'M1_IC_VCE_ACTIVITY',
                             'M2_OG_VCE_CNT_LOCAL',
                             'M2_OG_VCE_AMT_CHRG_LOCAL',
                             'M2_IC_VCE_OFFNET_CNT',
                             'M2_TOTL_MBS_ROAM',
                             'M2_TOTL_OG_SMS_ROAM',
                             'M2_TOTL_OG_PAYG_SMS_ROAM_AMT',
                             'M2_TOTL_OG_PKG_SMS_ROAM',
                             'M2_OG_SMS_OFFNET',
                             'M2_IC_VCE_ACTIVITY',
                             'M3_OG_VCE_DUR',
                             'M3_OG_VCE_DUR_ROAM',
                             'M3_PKG_OG_VCE_DUR_LOCAL',
                             'M3_PAYG_OG_VCE_DUR',
                             'M3_OG_VCE_CNT_ROAM',
                             'M3_PKG_OG_VCE_CNT_LOCAL',
                             'M3_PAYG_OG_VCE_CNT_ROAM',
                             'M3_OG_VCE_AMT_CHRG',
                             'M3_OG_VCE_AMT_CHRG_LOCAL',
                             'M3_OG_VCE_ONNET_CNT',
                             'M3_OG_VCE_OFFNET_CNT',
                             'M3_IC_VCE_CNT_LOCAL',
                             'M3_IC_VCE_AMT_CHRG',
                             'M3_IC_VCE_AMT_CHRG_LOCAL',
                             'M3_PAYG_TOTL_MBS_ROAM',
                             'M3_PAYG_TOTL_MBS_ROAM_AMT',
                             'M3_TOTL_OG_SMS_ROAM',
                             'M3_TOTL_OG_PAYG_SMS_ROAM_AMT',
                             'M3_TOTL_OG_PKG_SMS_ROAM',
                             'M3_IC_VCE_ACTIVITY',
                             'W1_OG_VCE_DUR',
                             'W1_OG_VCE_DUR_MIDNIGHT',
                             'W1_OG_VCE_DUR_LOCAL',
                             'W1_OG_VCE_CNT_ROAM',
                             'W1_IC_VCE_DUR_ROAM',
                             'W1_IC_VCE_CNT',
                             'W1_IC_VCE_CNT_LOCAL',
                             'W1_TOTL_MBS_MORNING',
                             'W1_OG_VCE_ACTIVITY',
                             'W1_IC_VCE_ACTIVITY',
                             'W1_DATA_ACTIVITY',
                             'W1_OG_SMS_ACTIVITY',
                             'W2_PKG_OG_VCE_DUR',
                             'W2_PKG_OG_VCE_DUR_LOCAL',
                             'W2_PKG_OG_VCE_CNT_ROAM',
                             'W2_IC_VCE_CNT_ROAM',
                             'W2_PAYG_TOTL_MBS',
                             'W2_PAYG_TOTL_MBS_AMT',
                             'W2_PAYG_TOTL_MBS_ROAM_AMT',
                             'W2_IC_VCE_ACTIVITY',
                             'W2_DATA_ACTIVITY',
                             'W4_TOTL_OG_SMS_ROAM',
                             'W4_TOTL_OG_PAYG_SMS_ROAM_AMT',
                             'W4_TOTL_OG_PKG_SMS_ROAM',
                             'W4_MONTHLY_PACKAGES',
                             'W4_ANNUAL_PACKAGES',
                             'W4_SUBSCRIPTION_CHARGE',
                             'W4_TOTAL_VOICE_RESOURCE',
                             'W4_TOTAL_SMS_RESOURCE',
                             'W4_TOTAL_GPRS_RESOURCE',
                             'W4_BUNDLE_VOICE_ONLY',
                             'W4_BUNDLE_SMS_ONLY',
                             'W4_BUNDLE_DATA_ONLY',
                             'W4_BUNDLE_HYBRID',
                             'W4_VCE_RES_TOTAL',
                             'W4_VCE_RES_REMAIN',
                             'W4_VCE_RES_USED',
                             'W4_SMS_RES_TOTAL',
                             'W4_SMS_RES_REMAIN',
                             'W4_SMS_RES_USED',
                             'W4_GPRS_RES_TOTAL',
                             'W4_GPRS_RES_REMAIN',
                             'W4_GPRS_RES_USED',
                             'W4_TOPUP_AMOUNT',
                             'W4_TOPUP_360',
                             'W4_TOPUP_VOUCHERRECHARGE',
                             'W4_TOPUP_BUNDLEBYSMS',
                             'W4_TOPUP_PAY360',
                             'W4_TOPUP_CARD',
                             'W4_TOPUP_CREDIT_CARD',
                             'W4_TOPUP_NDLPAYPAL',
                             'W4_TOPUP_MANUAL',
                             'W4_TOPUP_CASH',
                             'W4_PKG_SUBSCRIPTION_ACTIVITY',
                             'W4_TOPUP_ACTIVITY'
                             ]

        # Create a new dataframe with only the selected features
        df_selected = X_Normalized[selected_features]
        logging.info("Feature Engineering Completed.")
    else:
        logging.info("NO, Feature Engineering is OFF.")
        logging.info("Considering all columns as selected features.")
        selected_features = X_Normalized.columns
        df_selected = X_Normalized[selected_features]
    # --------------------------------------------------------------------------------------------------------------
    # Get the current script directory
    # script_dir = os.path.dirname(__file__)

    # Construct the path to the model file
    OUTPUT_path = os.path.join(OUTPUT_dir, Project_Name + '_SelectedFeatures.txt')

    logging.info("Creating SelectedFeatures.txt & writing selected features in it.")
    # Write the column names to a text file
    with open(OUTPUT_path, 'w') as file:
        for column in selected_features:
            file.write(column + '\n')

    logging.info("Created & Written.")

    # --------------------------------------------------------------------------------------------------------------
    # print('df_selected:', len(df_selected))
    # print('df[MSISDN]:', len(df['MSISDN']))

    df_selected.index = df.index

    # Add the target column and Customer_ID back to the dataframe
    # df_selected[Customer_ID] = ID
    df_selected = pd.concat([df['MSISDN'], df_selected], axis=1)
    df_selected = pd.concat([df_selected, df['FLAG']], axis=1)
    # df_selected[target_column] = y

    return df_selected.reset_index(drop=True)
