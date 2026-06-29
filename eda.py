import pandas as pd

df = pd.read_csv("data/train.csv")

print(df.shape)
print(df.info())
print(df.isnull().sum().sort_values(ascending=False).head(20))
cols_fill_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                   'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                   'MasVnrType']

for col in cols_fill_none:
    df[col] = df[col].fillna('None')
df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])
print(df.isnull().sum().sum())
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice'], kde=True)
plt.title('Распределение цен на жильё')
plt.savefig('images/price_distribution.png')
plt.close()

print("Skew (скошенность):", df['SalePrice'].skew())
df['SalePrice_log'] = np.log1p(df['SalePrice'])

print("Skew после логарифмирования:", df['SalePrice_log'].skew())

plt.figure(figsize=(8,5))
sns.histplot(df['SalePrice_log'], kde=True)
plt.title('Распределение цен на жильё (после log)')
plt.savefig('images/price_distribution_log.png')
plt.close()
df_encoded = pd.get_dummies(df, drop_first=True)
print(df_encoded.shape)
X = df_encoded.drop(['SalePrice', 'SalePrice_log', 'Id'], axis=1)
y = df_encoded['SalePrice_log']

print(X.shape, y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression — RMSE: {rmse_lr:.4f}, R2: {r2_lr:.4f}")
# Примерная ошибка в долларах (грубая оценка)
example_price = 200000  # условный дом за 200k
log_price = np.log1p(example_price)
error_high = np.expm1(log_price + rmse_lr)
error_low = np.expm1(log_price - rmse_lr)
print(f"Для дома в $200,000 типичный разброс предсказания: ${error_low:.0f} - ${error_high:.0f}")
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest — RMSE: {rmse_rf:.4f}, R2: {r2_rf:.4f}")
from xgboost import XGBRegressor

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost — RMSE: {rmse_xgb:.4f}, R2: {r2_xgb:.4f}")
import joblib

joblib.dump(xgb_model, 'models/house_price_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_columns.pkl')

print("Модель сохранена!")
