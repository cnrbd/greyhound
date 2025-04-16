import pandas as pd
import os
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xg
import pickle

def preprocess_dataframe(df):
    """
    Preprocesses the given DataFrame by:
    1. Dropping the 'comment1' column.
    2. Applying one-hot encoding to the 'stadium' column with a prefix.
    3. Converting 'birthdate', 'date1', and 'date2' to datetime.
    4. Calculating 'day_age_1', 'day_age_2', and 'days_between_race'.
    5. Dropping the 'birthdate', 'date1', and 'date2' columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Drop the 'comment1' column
    df = df.drop(columns=["comment1"])

    # One-hot encode the 'stadium' column
    df = pd.get_dummies(df, prefix="stadium_", columns=["stadium"])

    # Convert date columns to datetime
    df['birthdate'] = pd.to_datetime(df['birthdate'])
    df['date1'] = pd.to_datetime(df['date1'])
    df['date2'] = pd.to_datetime(df['date2'])

    # Calculate new features based on date differences
    df["day_age_1"] = (df['date1'] - df['birthdate']).dt.days
    df["day_age_2"] = (df['date2'] - df['birthdate']).dt.days
    df["days_between_race"] = (df['date2'] - df['date1']).dt.days

    # Drop the original date columns
    df = df.drop(columns=["birthdate", "date1", "date2"])

    return df


filepath = os.path.join(os.getcwd(), "df.csv")
train_file = pd.read_csv(filepath)
train_file = preprocess_dataframe(train_file)

X_train_file = train_file.drop(columns = ["time2"])
y_train_file = train_file["time2"]

#Scaling features for the model
scaler = StandardScaler()
scaler.fit(X_train_file)
X_scaled = scaler.transform(X_train_file)

#Use XGBoost model to predict the time2
best_xg = xg.XGBRegressor(booster = "gbtree", eta = 0.15, reg_lambda = 1.5, eval_metric = "rmse",n_estimators = 350 )
best_xg.fit(X_scaled, y_train_file)

#Prediction on the unseen csv file
filepath_unseen = filepath = os.path.join(os.getcwd(), "unseendf_example.csv")
x2 = pd.read_csv(filepath_unseen)

x2 = preprocess_dataframe(x2)

#One hot encoding messes up the Scaler object "fitting", so fill in missing binary stadium columns with 0
missing_cols = set(X_train_file.columns) - set(x2.columns)
for col in missing_cols:
    x2[col] = 0


#Ensure the columns are in the same order as the training data
x2 = x2[X_train_file.columns]
x2_scaled = scaler.transform(x2)
x2['predtime'] = best_xg.predict(x2_scaled)

x2.to_csv("~/Downloads/mypred.csv", index=False)

#Pickle!
with open("best_xg.pkl", "wb") as f:
    pickle.dump(best_xg, f)