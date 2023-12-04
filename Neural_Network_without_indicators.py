# Neyronika diktya

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from matplotlib import pyplot as plt

# Λήψη δεδομένων μετοχής από το yfinance
symbol = 'AMZN'  # Παράδειγμα με μετοχή Amazon
start_date = '2018-11-30'
end_date = '2023-11-30'
data = yf.download(symbol, start=start_date, end=end_date)

# Επιλογή τιμής κλεισίματος ως χαρακτηριστικό
features = data[['Close']]

# Κλιμακοποίηση των χαρακτηριστικών στο εύρος [0, 1]
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Δημιουργία συνόλων εκπαίδευσης και ελέγχου
X, y = [], []
look_back = 10  # Παράδειγμα με παραθυροσχήματα μήκους 10
for i in range(len(features_scaled) - look_back):
    X.append(features_scaled[i:i+look_back])
    y.append(features_scaled[i+look_back])

X, y = np.array(X), np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Δημιουργία και εκπαίδευση του μοντέλου νευρωνικού δικτύου
model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Προβλέψεις στα δεδομένα ελέγχου
predictions_scaled = model.predict(X_test)
predictions = scaler.inverse_transform(predictions_scaled)

# Αξιολόγηση του μοντέλου
mse = mean_squared_error(features[-len(y_test):], predictions)
print(f'Mean Squared Error: {mse}')

# Παρουσίαση των πραγματικών τιμών και των προβλέψεων
plt.plot(features.index[-len(y_test):],
         features[-len(y_test):], label='Πραγματικές τιμές')
plt.plot(features.index[-len(y_test):], predictions, label='Προβλέψεις')
plt.xlabel('Ημερομηνία')
plt.ylabel('Τιμή Κλεισίματος')
plt.title('Πραγματικές τιμές vs Προβλέψεις')
plt.legend()
plt.show()
