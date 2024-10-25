
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import pickle
import pandas as pd


def predictPrice(time_period):

    time_period = int(time_period)
    model_path = "TrainedModels/AvocadoPricePrectionModel.pkl"
    model = pickle.load(open(model_path, "rb"))
    
    today = date.today()
    print(today)
    format = "%Y-%m-%d"

    today = today.strftime(format)

    future_dates = pd.date_range(today, periods=time_period, freq='m')
    forecast_results = model.get_forecast(steps=time_period)
    future_predictions = forecast_results.predicted_mean
    forecast_df = pd.DataFrame({'date': future_dates, 'prediction': future_predictions})
    
    date_prediction_dict = {}

    for index, row in forecast_df.iterrows():
        current_date = row['date']
        current_date = current_date.strftime(format) 
        prediction = row['prediction']
        date_prediction_dict[current_date] = prediction


    return (date_prediction_dict)
    
