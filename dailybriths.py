code
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
# import fbprophet
# from fbprophet.plot import add_changepoints_to_plot
# import prophet
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot, plot_plotly
import warnings
import matplotlib.pyplot as plt
df = pd.read_csv("/Users/vinay/Downloads/DBF/daily-total-female-births-
                 CA.csv",  parse_dates=['date'], date_parser=pd.to_datetime)
df.columns = ['ds', 'y']
df.head()
plt.figure(figsize=(20,10))
plt.plot(df['ds'], df['y']);
plt.grid()
plt.title('Daily Female Births in 1959')
plt.show()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m =Prophet(yearly_seasonality=True, daily_seasonality=False, 
                          changepoint_range=0.9, 
                          changepoint_prior_scale=0.5,
                          seasonality_mode='multiplicative')
    m.fit(df)
future = m.make_future_dataframe(periods=90, freq='d')
future.tail()
forecast = m.predict(future)
forecast.tail()
fig2=m.plot_components(forecast)
fig2.show()
fig3=m.plot(forecast)
fig3.show()
#from fbprophet.plot import plot_plotly
from prophet.plot import add_changepoints_to_plot, plot_plotly
#let's plot the forecast
plot_plotly(m, forecast)