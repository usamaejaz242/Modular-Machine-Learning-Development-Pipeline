import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
import json
with open("config.json", "r") as f:
    config = json.load(f)



# ---------------------------- Utility ----------------------------

def reshape_data(x):
    return np.reshape(x.values, (x.shape[0], 1, x.shape[1]))


# ---------------------------- Model Trainers ----------------------------

def TrainXGBoost(X_train, y_train, logging):
    logging.info("Initializing Hyperparameters for XGBoost.")
    
    model = XGBClassifier(
        colsample_bytree=0.8,
        gamma=0.6,
        learning_rate=0.2,
        max_depth=10,
        n_estimators=122,
        reg_alpha=0.01,
        reg_lambda=1,
        subsample=0.8,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    logging.info("Training XGBoost model.")
    model.fit(X_train, y_train)
    logging.info("XGBoost training completed.")

    return model


def LSTM_model(X_train, X_test, y_train, y_test):
    X_train = reshape_data(X_train)
    X_test = reshape_data(X_test)

    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    return model, X_test


def rnn_model(X_train, X_test, y_train, y_test):
    X_train = reshape_data(X_train)
    X_test = reshape_data(X_test)

    model = Sequential([
        SimpleRNN(50, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    return model, X_test


def nn_model(X_train, X_test, y_train, y_test):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])    
    early_stop = EarlyStopping(
        monitor='val_loss',     
        patience=10,             
        restore_best_weights=True
    )


    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),callbacks=[early_stop], verbose=1)

    return model


# ---------------------------- Model Preparation ----------------------------

def ModelPreperation(FinalData, custom_threshold, current_date, logging, Project_Name, OUTPUT_dir):
    logging.info("Starting Model Preparation...")

    # --- Step 1: Separate Features and Target ---
    target_column = 'FLAG'
    X = FinalData.drop(columns=[target_column, 'MSISDN'], errors='ignore')
    y = FinalData[target_column]

    # --- Step 2: Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------ SELECT ONE MODEL TO USE ------------------
    if config.get("Run_XGBoost", 0):
        logging.info("Running XGBoost model.")
        model = TrainXGBoost(X_train, y_train, logging)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

    elif config.get("Run_LSTM", 0):
        logging.info("Running LSTM model.")
        model, X_test = LSTM_model(X_train, X_test, y_train, y_test)
        y_pred_prob = model.predict(X_test).flatten()

    elif config.get("Run_RNN", 0):
        logging.info("Running RNN model.")
        model, X_test = rnn_model(X_train, X_test, y_train, y_test)
        y_pred_prob = model.predict(X_test).flatten()

    elif config.get("Run_NN", 0):
        logging.info("Running Neural Network model.")
        model = nn_model(X_train, X_test, y_train, y_test)
        y_pred_prob = model.predict(X_test).flatten()

    else:
        logging.warning("No model flag is set to 1 in config. Exiting.")
        return
    
    model = TrainXGBoost(X_train, y_train, logging)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    # ------------------ THRESHOLD & METRICS ------------------

    y_pred = (y_pred_prob >= custom_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # logging.info(f"Accuracy: {acc:.4f}")
    # logging.info(f"Classification Report:\n{report}")
    # logging.info(f"Confusion Matrix:\n{cm}")

    # ------------------ SAVE MODEL ------------------

    model_save_path = os.path.join(OUTPUT_dir, f"{Project_Name}_Model")

    if isinstance(model, tf.keras.Model):
        model.save(model_save_path + ".h5")
        logging.info("Keras model saved successfully.")
    else:
        joblib.dump(model, model_save_path + ".pkl")
        logging.info("Scikit-learn model saved successfully.")

    # ------------------ SAVE PREDICTIONS ------------------

    pred_df = pd.DataFrame({
        'True': y_test.values,
        'Predicted': y_pred,
        'Probability': y_pred_prob
    })

    pred_path = os.path.join(OUTPUT_dir, f"{Project_Name}_Predictions_{current_date}.csv")
    pred_df.to_csv(pred_path, index=False)
    logging.info(f"Predictions saved to {pred_path}")

    logging.info("Model preparation and prediction completed.")
