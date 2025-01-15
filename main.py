import streamlit as st
import yfinance as yf
from prophet import Prophet # modelimiz
from prophet.plot import plot_plotly
from plotly import graph_objs as go # etkileşimli grafikler için
from datetime import date
import numpy as np
np.float_ = np.float64





START = "2020-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


# Şimdi web uygulamamızı yapmaya başlayabiliriz
st.title("Hisse Senedi Tahmin Uygulaması")

stocks = ("AAPL" , "GOOG" , "MSFT" , "GME" , "CCOLA.IS" , "ASELS.IS" , "KCHOL.IS" , "SAHOL.IS" , "AKBNK.IS")    # tahminde bulunacağımız hisselerin yahoo'daki kodları
selected_stock = st.selectbox("Tahmin için veri seti seçin" , stocks)  # hisseleri seçebilmek için selectbox yaptık
n_years = st.slider("Tahmin yılı: " , 1 , 4)    # yılı seçebilmek için de slider yaptık (1 ile 4 yıl arasında bir seçim yapabilecek)
period = n_years * 365










@st.cache_data  # indirilen verileri sürekli indirmemsi için cache'de tutacağız
# Hisse verilerini yükleyeceğiz
def load_data(stock_name):
    data = yf.download(stock_name , START , TODAY)  # girilen hissenin başlangıçtan bugüne kadar verisi yükeleyecek yahoo finance'tan
    data.reset_index(inplace=True)  # tarihi ilk sütuna koyabilmek için
    return data

# Yahoo'dan verileri yüklerken ekrana bir çıktı verdik
data_load_state = st.text("Veri yükleniyor... ")
data = load_data(selected_stock)
data_load_state.text("Veri yüklenmesi tamamlandı...")


st.subheader("Hissenin Şuanki Verilerine Göz At")
st.write(data.tail())   # en güncel verileri başta görmek istediğimiz için tail yazdık head yerine








# Verileri görselleştirmek için bir fonksiyon yazıyoruz
def plot_raw_data():
    fig = go.Figure()
    fig.layout.update(title_text="Zaman Serisi Verileri" , xaxis_rangeslider_visible=True)
    fig.add_trace(go.Scatter(x=data["Date"] , y=data["Open"] , name="açılış fiyatı"))
    fig.add_trace(go.Scatter(x=data["Date"] , y=data["Close"] , name="kapanış fiyatı"))
    fig.update_layout(xaxis_title="Tarih", yaxis_title="Fiyat")
    st.plotly_chart(fig)    # plotky_chart ile birlikte etkileşimli grafik çizdirebiliyoruz.

plot_raw_data()









# **MODEL VE TAHMİN**

# Şimdi tahmin işlemimizi gerçekleştireceğiz. 
# Bunun için Facebook'un geliştirmiş olduğu "prophet" algoritmasını kullanacağız
# Prophet modeli, bir zaman serisi tahmin modeli olup, burada hisse senedi fiyatlarını tahmin etmek için kullanacağız.

df_train = data[["Date" , "Close"]]    # bunu yapabilmek için önce veriyi belirli bir formata çevirmeliyiz
df_train = df_train.rename(columns={"Date":"ds" , "Close":"y"})    # sütunları yeniden adlandırmalıyız çünkü Facebook prophet bunu belirli bir formatta bekliyor.


m = Prophet() 
m.fit(df_train) # Model
future = m.make_future_dataframe(periods=period) # belirli bir zaman dilimi için tahmin yapmak üzere bir veri çerçevesi (dataframe) oluşturduk.
tahmin = m.predict(future)  # Tahmin

st.subheader("Tahmin Verileri")
st.write(tahmin.tail())  

# Şimdi üstteki tahmin verileri için de bir görsel oluşturalım
fig2 = plot_plotly(m , tahmin)
fig2.update_layout(xaxis_title="Tarih", yaxis_title="Fiyat")
st.plotly_chart(fig2)


st.subheader("Tahmin Bileşenleri")
fig3 = m.plot_components(tahmin)
st.write(fig3)






# conda activate stock_prediction
# streamlit run main.py