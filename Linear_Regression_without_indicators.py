# Αλγόριθμος Γραμμικής Παλινδρόμησης

# Εισαγωγή απαραίτητων βιβλιοθηκών
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Λήψη δεδομένων από το χρηματιστήριο
symbol = 'AAPL'  # Παράδειγμα με μετοχή Apple
start_date = '2018-11-30'
end_date = '2023-11-30'
df = yf.download(symbol, start=start_date, end=end_date)


# Καθαρισμός και επεξεργασία δεδομένων
df.dropna(inplace=True)  # Διαγραφή απουσιάζουσων τιμών
df['Return'] = df['Close'].pct_change()  # Υπολογισμός ημερήσιας απόδοσης
df = df.reset_index()


# Οπτικοποίηση δεδομένων
plt.figure(figsize=(12, 6))
# Μετατροπή της στήλης Date σε datetime
df['Date'] = pd.to_datetime(df['Date'])
plt.plot(df['Date'], df['Close'], label='Close Price')  # Τιμή κλεισίματος
plt.title('Historical Close Price of AAPL')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.gcf().autofmt_xdate()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
plt.gca().xaxis.set_tick_params(which='major', pad=15)

plt.show()

# Δημιουργία μεταβλητών για το μοντέλο μηχανικής μάθησης
# Τιμή κλεισίματος προηγούμενης ημέρας
df['Previous_Close'] = df['Close'].shift(1)
df.dropna(inplace=True)  # Διαγραφή απουσιάζουσων τιμών μετά τη μετατόπιση

# Διαχωρισμός σε σετ εκπαίδευσης και δοκιμής
X = df[['Date', 'Previous_Close']]
y = df['Close']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False)

# Εκπαίδευση του μοντέλου
model = LinearRegression()
model.fit(X_train.drop('Date', axis=1), y_train)

# Πρόβλεψη τιμών
predictions = model.predict(X_test.drop('Date', axis=1))

# Αξιολόγηση του μοντέλου
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f'MSE: {mse}')
print(f'MAE: {mae}')

plt.figure(figsize=(12, 6))
plt.plot(X_test['Date'], y_test, color='blue',
         label='Actual Price')  # Actual prices
plt.plot(X_test['Date'], predictions, color='orange',
         label='Predicted Price')  # Predicted prices
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.gcf().autofmt_xdate()  # Auto-rotate dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(
    mdates.MonthLocator(interval=6))  # Set ticks every 6 months

plt.show()
