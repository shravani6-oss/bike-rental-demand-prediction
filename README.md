# Bike Rental Demand Prediction with Automatic Weather Inputs and a Prediction GUI

## Project Overview
This project predicts the hourly demand for rental bikes using Machine Learning regression models. It uses historical bike-sharing data and features such as season, hour, temperature, humidity, weather conditions, and working-day information to estimate the total number of bike rentals.

The project also provides a user-friendly Graphical User Interface (GUI) where users can enter details, select a city, retrieve weather information, and obtain predicted bike rental demand.

## Objectives
- Analyze historical bike rental data.
- Perform data preprocessing and exploratory data analysis.
- Train and compare multiple regression models.
- Evaluate model performance using MAE, RMSE, and R² score.
- Build an interactive GUI for bike rental demand prediction.
- Integrate weather information using the Open-Meteo API.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Requests
- **GUI:** Streamlit
- **Dataset:** UCI Bike Sharing Dataset
- **Weather API:** Open-Meteo

## Machine Learning Models
The project compares the following regression algorithms:
1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Hist Gradient Boosting Regressor

The best-performing model is selected based on the evaluation results.

## Dataset
The project uses the UCI Bike Sharing Dataset, which contains historical bike rental records and information about weather and time.

Dataset source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

## Project Structure
```text
bike-rental-demand-prediction/
├── Bike_Rental_Demand_Prediction_Colab_GUI.ipynb
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── best_bike_model.joblib
    ├── bike_features.joblib
    └── model_comparison.csv
```

*Note: The artifacts folder and its files are generated after running the training notebook.*

## Installation and Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/bike-rental-demand-prediction.git
cd bike-rental-demand-prediction
```

Replace `YOUR-USERNAME` with your actual GitHub username.

### 2. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application
Ensure that the trained model files are available in the `artifacts` folder, then execute:

```bash
streamlit run app.py
```

The application will open in your web browser.

## Features
- Historical bike rental data analysis
- Comparison of multiple regression models
- Automatic weather data retrieval
- Interactive GUI for entering prediction details
- Hourly bike rental demand prediction
- Model evaluation and performance comparison

## Evaluation Metrics
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R² Score:** Coefficient of Determination

Actual model evaluation results should be added after running the notebook.

## Future Scope
- Improve prediction accuracy using additional data.
- Integrate historical weather data for date-specific predictions.
- Deploy the application online.
- Add interactive charts and demand trend visualizations.
- Extend predictions to different cities and bike-sharing systems.

## Conclusion
This project demonstrates the application of Machine Learning regression techniques to estimate bike rental demand. By combining historical data, weather information, and an interactive GUI, it provides a practical approach to understanding and forecasting hourly bike usage.

## Author
**Student Name:** Shravani Kali  
**Course:** B.Tech – Computer Science and Engineering (AI & Data Science)  
**Institution:** Ramdeobaba University, Nagpur  
**Academic Year:** 2026–2027

## License
This project is intended for educational and academic purposes.
