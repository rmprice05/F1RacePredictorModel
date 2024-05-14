Formula 1 Race Outcome Prediction Model
Project Overview
This repository contains a predictive model for Formula 1 race outcomes. It uses historical race and qualifying data from 1980 up to the last completed year, fetched via Ergast API. The model predicts whether a driver will win a race based on the integrated data set.

Data Collection: Uses requests to fetch data from the Ergast API.
Preprocessing: Includes merging datasets, handling categorical variables, imputing missing values, and scaling.
Modeling: Logistic Regression used for prediction after training on historical data.
Evaluation: Includes accuracy assessment and potential areas for model improvement.

Future Work:
Explore more complex models like Random Forests and Gradient Boosting Machines for potential accuracy improvements.
Implement additional feature engineering techniques to capture more dynamics of the race outcomes, such as DNF rates, circuit types, and weather.
