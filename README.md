# Intelligent Demand Forecasting Agent

An AI-powered system that predicts product demand from historical sales data and 
recommends inventory actions to prevent overstocking and stockouts.

## Features
- Historical sales analysis across 10 stores × 50 products
- Time-series decomposition (trend + seasonality)
- Statistical trend significance testing (Mann-Kendall)
- Anomaly detection and cleaning
- Demand forecasting with prediction intervals (Holt-Winters, validated ~22.5% avg error across sampled products)
- EOQ-based inventory recommendation (reorder point, safety stock, optimal order quantity)
- Fleet-wide risk dashboard scanning all products for stockout risk
- What-if demand simulation

## Tech Stack
Python, pandas, statsmodels, pymannkendall, Streamlit, Plotly

## Run locally
