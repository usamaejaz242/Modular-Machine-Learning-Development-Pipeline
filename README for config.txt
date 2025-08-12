1. For Training, set "TrainingMode" to 1 and for Testing/Deployment set "TrainingMode" to 0 or any number other than 1.
2. Set "SQL_Data"to 1, to use THM_DAILY_KPIS. If it is set to 1 then the csv file will not be read.
3. Set "Feature_Engineering" to 1, to use feature selection. Otherwise all the features will be used for model training and deployment.
4. Also, provide server for server name, database for db name, username and password.
5. Make sure the VPN is connected. Otherwise no data will be fetched. For example:

    "server": "100.100.120.117",
    "database": "dd_ds",
    "username": "usama_abdullah",
    "password": "usama@$%^"

6. If you want to play arounf with dummy datsets, there are currently 3 datasets used for development of this pipeline.
	6.1 For using old TalkHome Mobile data, set "CSV_Flag" to 1.
	6.2 For using publicly available telco data from Kaglle, set "CSV_Flag" to 2.
	6.3 For using Dummy data, set "CSV_Flag" to any number other than 1 and 2 (i.e, 0,3,4,5,...).

NOTE: First Time Training Must Be Performed Before Using AI Model For Deployment.