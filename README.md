# Two-Stage Event Success Prediction Framework
A business case study for predicting event success before publication and optimizing marketing investments after launch.
## Project Overview
Marketing decisions evolve over time.
This repository presents a two-stage machine learning framework designed to support event planning throughout the early lifecycle of an event.
> The first model (T0) predicts event success before publication using only static event information.
> The second model (T+1 Week) updates the prediction using user engagement metrics collected during the first week after publication, enabling dynamic marketing optimization.
Together, the two models demonstrate how predictive analytics can support strategic decision-making before and after an event is launched.
## Business Problem
Launching an unsuccessful event generates unnecessary marketing costs and inefficient resource allocation.
The objective of this project is to support decision-making before publication by estimating the probability that an event will be successful, using the resulting predictions to optimize advertising budgets, prioritize marketing campaigns and improve operational planning.
## Business Questions
This analysis aims to answer the following business questions:
- Which events are most likely to succeed?
- Which variables have the greatest impact on event success?
- Which cities and event categories perform best?
- Can machine learning improve marketing decisions before publication?
- How can predicted success scores support budget allocation?
- How can the model be transformed into a practical decision-support tool?

## Stage 1 — Pre-Publication Prediction (T0)
Objective
Predict event success before publication using only static event information.
> ✔ Accuracy: 70.9%
> ✔ ROC-AUC: 0.73
> ✔ Decision: Should this event be promoted?

📓 Notebook:
01_Pre_Publication_Prediction.ipynb

## Stage 2 — Post-Publication Optimization (T+1)
Objective
Update the prediction after one week using user engagement metrics
> ✔ Accuracy: 84.2%
> ✔ ROC-AUC: 0.92
> ✔ Decision: Should marketing investment increase or be focused only on defined events?

📓 Notebook:
02_Post_Publication_Optimization.ipynb

## Model comparison
|                   | T0             | T+1              |
| ----------------- | -------------- | ---------------- |
| Information       | Static         | Behaviour        |
| Accuracy          | 70.9%          | 84.2%            |
| ROC-AUC           | 0.73           | 0.92             |
| Business Decision | Publish?       | Increase Budget? |
| Main Value        | Prioritization | Optimization     |

## Decision Support Dashboard
## Business Impact
The proposed framework supports marketing decision-making throughout two critical stages:
### Before publication
• Estimate event potential
• Prioritize campaigns
• Allocate initial advertising budgets
### After one week
• Re-evaluate event performance
• Increase or reduce investments
• Optimize ongoing campaigns
Together, the two models transform historical event data into a practical decision-support framework for marketing managers.


## Key Takeaway
Rather than developing a single prediction model, this project demonstrates how machine learning can support marketing decisions throughout the lifecycle of an event.
The proposed two-stage framework combines pre-publication forecasting with post-publication optimization, illustrating how predictive analytics can evolve from a simple classification model into a practical business decision-support system.





