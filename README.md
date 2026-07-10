# Predicting Event Success for Data-Driven Marketing Decisions
This project analyzes historical data from an Italian company organizing social events and experiences. The goal is to predict whether an event will succeed or fail using only information available at publication time (T0).
## Project Overview
Marketing teams frequently invest advertising budgets without knowing whether an event is likely to succeed.
This project develops a machine learning model capable of predicting event success using only the information available at publication time (T0).
Rather than simply building a classifier, the project demonstrates how predictive analytics can support marketing planning, reduce investment risk and prioritize high-potential events before launch.
## Business Problem
Launching an unsuccessful event generates unnecessary marketing costs and inefficient resource allocation.
The objective of this project is to support decision-making before publication by estimating the probability that an event will be successful, using the resulting predictions to optimize advertising budgets, prioritize marketing campaigns and improve operational planning.
## Business Questions
This analysis aims to answer the following business questions:
• Which events are most likely to succeed?
• Which variables have the greatest impact on event success?
• Which cities and event categories perform best?
• Can machine learning improve marketing decisions before publication?
• How can predicted success scores support budget allocation?
• How can the model be transformed into a practical decision-support tool?
## Key Results
| KPI | Result |
|------|---------|
| Dataset | 666 historical events |
| Target | Event Success |
| Model | Logistic Regression |
| Accuracy | 76% |
| ROC-AUC | 0.70 |
| Business Goal | Marketing Budget Optimization |
| Final Output | Event Success Probability |
## Data Preparation
Only variables available before event publication (T0) were retained.
Variables describing user behaviour after publication (such as one-week visitors and reading time evolution) were removed to simulate a realistic business scenario.
This prevents data leakage and ensures that predictions can be used operationally before launching an event.

```python
# import libraries:
%matplotlib inline
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_curve
from IPython.core.display import HTML as Center
```
```python
#Centering the plots
Center(""" <style>
.jp-RenderedImage {
    display: table-cell;
    text-align: center;
    vertical-align: middle;
}
</style> """)
```
```python
# Seaborn visualization settings:
sns.set_theme(
    style="darkgrid",      
    context="notebook",        
    palette="deep"
)
```
```python
# Load cleaned dataset
df = pd.read_excel("anonimized_data.xlsx")

# Remove variables unavailable at publication time
df_t0 = df.drop(columns=[
    "VISITATORI + 1 WEEK",
    "DELTA VISITATORI",
    "TEMPO DI LETTURA",
    "TEMPO + ONE WEEK"
])

df_t0.head()
```
| Day | Month | Visitors | City | Age Group | Current Bookings | Conversion Rate | Event Category | Region | Year | Success |
|:----|:------|---------:|:-----|:----------|-----------------:|----------------:|:---------------|:-------|-----:|--------:|
| Day_4 | Month_1 | 4  | City_1 | Age_Group_2 | 0 | 0.0000 | Category_4 | North   | 2023 | 0 |
| Day_4 | Month_1 | 4  | City_2 | Age_Group_2 | 5 | 0.0602 | Category_8 | Central | 2022 | 0 |
| Day_5 | Month_3 | 45 | City_2 | Age_Group_2 | 3 | 0.0566 | Category_4 | Central | 2022 | 0 |
| Day_5 | Month_2 | 21 | City_2 | Age_Group_2 | 2 | 0.0667 | Category_4 | Central | 2022 | 0 |
| Day_3 | Month_2 | 88 | City_2 | Age_Group_2 | 3 | 0.0337 | Category_4 | Central | 2022 | 0 |
> **Target Variable**
>
> **Success**
> - **1** → Successful event
> - **0** → Unsuccessful event
The cleaned dataset contains only information available at the time of publication (T0), ensuring that the machine learning model predicts event success without using future information and avoiding data leakage.
## Exploratory Data Analysis
### Target Distribution
```python
target_distribution = (
    df_t0["STATO"]
    .value_counts(normalize=True)
    .rename_axis("Outcome")
    .reset_index(name="Percentage")
)

display(target_distribution)
```
| Outcome | Percentage |
|:--------|-----------:|
| 🔴 Failure (0) | **86.91%** |
| 🟢 Success (1) | **13.09%** | 

> **Business Insight:** 
>Approximately 87% of historical events were unsuccessful.
>This strong imbalance reflects a realistic business scenario where successful events are relatively rare, making prediction significantly more challenging.
```python
### Top Performing Cities
```python
# Analyze historical success rate by city 
city_analysis = df_t0.groupby('CITTA')['STATO'].agg(['count', 'sum'])

city_analysis['success_rate'] = city_analysis['sum'] / city_analysis['count']

city_analysis = city_analysis.sort_values(by='success_rate', ascending=False)
city_analysis = city_analysis[city_analysis['count'] > 5]
city_analysis.head(10)
```
```python
# Top performing cities
plt.figure(figsize=(10,5))

top_cities = city_analysis.sort_values(
    by="success_rate",
    ascending=False
).head(10)

bars = plt.bar(
    top_cities.index,
    top_cities["success_rate"],
    edgecolor="black"
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.015,
        f"{bar.get_height():.2f}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Top 10 Cities by Historical Success Rate", fontsize=15, fontweight="bold")
plt.xlabel("City")
plt.ylabel("Historical Success Rate")
plt.ylim(0,1)
plt.xticks(rotation=45)
sns.despine()
plt.tight_layout()
plt.show()
```
![Top 10 Cities](event_files/Top_10_cities.png)
>  ***Business Insight:**
>Historical performance varies considerably across cities. Some locations consistently achieve higher success rates, suggesting that geographical context plays a significant role in event performance.
### Event Categories
```python
#  Best performing events
categoria_analysis = df_t0.groupby('CATEGORIA')['STATO'].agg(['count', 'sum'])

categoria_analysis['success_rate'] = categoria_analysis['sum'] / categoria_analysis['count']
categoria_analysis = categoria_analysis[categoria_analysis['count'] >= 10]
categoria_analysis.sort_values(by='success_rate', ascending=False)
```
| Event Category | Events | Successful Events | Success Rate |
|:---------------|-------:|------------------:|-------------:|
| 🥇 Category_4 | 353 | 61 | **17.28%** |
| 🥈 Category_3 | 152 | 20 | **13.16%** |
| 🥉 Category_9 | 77 | 8 | **10.39%** |
| Category_8 | 74 | 7 | **9.46%** |
| Category_11 | 135 | 7 | **5.19%** |
| Category_1 | 10 | 0 | **0.00%** |
>  **Business Insight:**
> Event performance varies considerably across categories. **Category_4** achieved the highest historical success rate (17.28%) while also representing the largest share of events, suggesting it offers the greatest marketing potential. Conversely, **Category_1** recorded no successful events during the observed period, indicating a low-priority category for future marketing investments.
### Days of the week performance
```python
# Best performing days
day_analysis = df_t0.groupby('GIORNO')['STATO'].agg(['count', 'sum'])

day_analysis['success_rate'] = day_analysis['sum'] / day_analysis['count']

day_analysis.sort_values(by='success_rate', ascending=False)
```
| Day of the Week | Events | Successful Events | Success Rate |
|:----------------|-------:|------------------:|-------------:|
| Day_4 | 271 | 41 | **15.13%** |
| Day_5 | 212 | 31 | **14.62%** |
| Day_7 | 199 | 26 | **13.07%** |
| Day_2 | 8 | 1 | **12.50%** |
| Day_3 | 84 | 7 | **8.33%** |
| Day_1 | 51 | 2 | **3.92%** |
>  **Business Insight**
> Event performance varies across the days of the week. **Day_4** achieved the highest historical success rate (15.13%), followed closely by **Day_5** and **Day_7**. In contrast, **Day_1** showed the lowest success rate (3.92%), suggesting that scheduling events on this day may require additional marketing efforts or a different promotional strategy.
## Feature Engineering
A historical city success rate was calculated exclusively on the training set.
This feature estimates the historical probability of success for each city while preventing data leakage.
Unknown cities are assigned the global average success rate observed in the training data.
```python
# Define features and target
features = [
    'CITTA',
    'CATEGORIA',
    'GIORNO',
    'MESE',
    'ZONA',
    'FASCIA DI ETA',
]

X = df_t0[features]
y = df_t0['STATO']

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Create historical city success rate feature
# This feature represents the historical probability of success for each city, calculated exclusively on the training set to avoid data leakage.
train_temp = X_train.copy()
train_temp['STATO'] = y_train

city_success = train_temp.groupby("CITTA")['STATO'].mean()

X_train['city_success_rate'] = X_train["CITTA"].map(city_success)
X_test['city_success_rate'] = X_test["CITTA"].map(city_success)

mean_rate = y_train.mean()

X_train['city_success_rate'] = X_train['city_success_rate'].fillna(mean_rate)
X_test['city_success_rate'] = X_test['city_success_rate'].fillna(mean_rate)

# Scale numerical features
scaler = StandardScaler()
num_cols = ['city_success_rate']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# One-hot encoding
X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Align train and test datasets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
```
## Machine Learning Model
A Logistic Regression classifier was selected because it provides transparent and interpretable predictions.
SMOTE was tested to handle class imbalance. However, results did not improve significantly. Due to the high number of categorical variables and the limited predictive information available at T0, synthetic samples generated by SMOTE  were not sufficiently representative. In particular, the recall of the success class decreased by approximately 11%, indicating a lower ability of the model to correctly identify successful events.The final model therefore uses: class_weight='balanced'
```python
# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)
```
```python
coeff = pd.Series(
    model.coef_[0],
    index=X_train.columns
)
```
## Model evaluation
```python
#  Generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
# Evaluation metrics
print("=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
```
```text
=== CLASSIFICATION REPORT ===
              precision    recall  f1-score   support

           0       0.95      0.68      0.80       148
           1       0.20      0.71      0.32        17

    accuracy                           0.68       165
   macro avg       0.58      0.69      0.56       165
weighted avg       0.88      0.68      0.75       165

Accuracy: 0.6848484848484848

ROC-AUC: 0.699523052464229

```
The model performs significantly better at identifying failed events than successful ones. This is expected given: strong class imbalance , limited information available at T0.  Despite moderate predictive power, the model remains useful as a business-support tool for identifying potentially promising events. ROC-AUC = 0.70 indicates moderate discriminative ability and demonstrates that the model provides useful decision support despite the limited information available at publication time.
Several historical success-rate features were tested, including city success rate, category success rate and combined city-category success rate. However, these variables did not produce significant improvements in model performance. This suggests that event success does not depend exclusively on structural factors such as geographical context or event category, but is also strongly influenced by more complex behavioral dynamics.
```python
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title( "Confusion Matrix - Logistic Regression",
    weight='bold',
    pad=12
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```
![matrix](event_files/conf_matrix.png)
```python
#  Feature importance analysis
top_features = coeff.sort_values().tail(8)
bottom_features = coeff.sort_values().head(8)

important_features = pd.concat([
    bottom_features,
    top_features
])

plt.figure(figsize=(10,6))

important_features.plot(kind='barh')

plt.title(
    "Most Influential Features in the Logistic Regression Model",
    weight='bold',
    pad=12
)
plt.xlabel("Logistic Regression Coefficient")
sns.despine()
plt.show()
```
![features](event_files/features.png)

```python

```
```python

```
```python

```

![Top 10 Cities](event_files/Top_10_cities.png)

