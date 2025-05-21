import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import joblib

def generate_data(n=1000, random_state=42):
    np.random.seed(random_state)  # For reproducibility

    # Generate realistic customer features
    age = np.random.normal(40, 12, n).clip(18, 80).astype(int)
    tenure_months = np.random.exponential(scale=12, size=n).astype(int)
    monthly_spend = np.random.normal(70, 30, n).clip(10, 200)

    # Calculate churn probability influenced by tenure and spend
    churn_prob = 0.5 - 0.03 * tenure_months + 0.002 * monthly_spend
    churn_prob = np.clip(churn_prob, 0, 1)

    # Generate churn labels based on probability
    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        'Age': age,
        'Tenure_Months': tenure_months,
        'Monthly_Spend': monthly_spend,
        'Churn': churn
    })

    print("Sample data:")
    print(df.head())

    return df

def train_churn_model(df):
    X = df[['Age', 'Tenure_Months', 'Monthly_Spend']]
    y = df['Churn']

    churn_model = RandomForestClassifier(n_estimators=100, random_state=42)
    churn_model.fit(X, y)

    joblib.dump(churn_model, 'churn_model.joblib')
    print("Churn model trained and saved.")

def train_revenue_forecast_model(df):
    # Use tenure and churn to predict monthly spend as a proxy for revenue forecasting
    X_rev = df[['Tenure_Months', 'Churn']]
    y_rev = df['Monthly_Spend']

    revenue_model = LinearRegression()
    revenue_model.fit(X_rev, y_rev)

    joblib.dump(revenue_model, 'revenue_forecast_model.joblib')
    print("Revenue forecasting model trained and saved.")

def main():
    df = generate_data()
    train_churn_model(df)
    train_revenue_forecast_model(df)

if __name__ == "__main__":
    main()
