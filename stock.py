import streamlit as st
from datetime import date
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
from prophet import Prophet
from plotly import graph_objs as go

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Market Prediction Web app ')

stocks = ('TSLS','AMZN','GOOG', 'AAPL', 'MSFT', 'GME')
selected_stock = st.selectbox('Select the stock for which prediction is to be done : ', stocks)

n_years = st.slider('select the years of prediction:', 1, 4)
period = n_years * 365


@st.cache
def load_data(ticker):
    data = yf.download(ticker,START,TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data =load_data(selected_stock)

data_load_state.text('Loading data... done!')
# st.write(data.columns)
# st.write(type(data))
st.subheader('Original Data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
plot_raw_data()

# fig = plt.figure(figsize=(10, 4))
# sns.countplot(x="open", data=data)
#
# st.pyplot(fig)
fig = plt.figure(figsize=(10,4))
sns.scatterplot(data=data,y=data['Open'],x=data['Date'])
st.pyplot(fig)

#forcasting data

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={'Date':'ds','Close':'y'})

m= Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forcast = m.predict(future)
st.write("Predicted data for the stock : ")
st.write(forcast.tail())
# st.write(forcast.columns)

fig2 = m.plot(forcast)
st.pyplot(fig2)

fig3 =m.plot_components(forcast)
st.pyplot(fig3)