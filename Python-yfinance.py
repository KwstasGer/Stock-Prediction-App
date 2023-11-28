#algorithmos grammikis palindromisis


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
#symbol = 'ALPHA.AT'
#data = yf.download(symbol, start='2020-01-01', end='2024-01-01')

symbol = 'AAPL'  # Παράδειγμα με μετοχή Apple
start_date = '2022-01-01'
end_date = '2023-01-01'
data = yf.download(symbol, start=start_date, end=end_date)

# Καθαρισμός και επεξεργασία δεδομένων
data.dropna(inplace=True) # Διαγραφή απουσιάζουσων τιμών
data['Return'] = data['Close'].pct_change() # Υπολογισμός ημερήσιας απόδοσης

# Οπτικοποίηση δεδομένων
plt.figure(figsize=(10, 6))
plt.plot(data['Close'], label='Κλείσιμο')
plt.title(f'Ιστορική Τιμή Κλεισίματος {symbol}')
plt.legend()
plt.show()

# Δημιουργία μεταβλητών για το μοντέλο μηχανικής μάθησης
data['Previous_Close'] = data['Close'].shift(1) # Τιμή κλεισίματος προηγούμενης ημέρας
data.dropna(inplace=True) # Διαγραφή απουσιάζουσων τιμών μετά τη μετατόπιση

# Διαχωρισμός σε σετ εκπαίδευσης και δοκιμής
X = data[['Previous_Close']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Εκπαίδευση του μοντέλου
model = LinearRegression()
model.fit(X_train, y_train)

# Πρόβλεψη τιμών
predictions = model.predict(X_test)

# Αξιολόγηση του μοντέλου
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print(f'MSE: {mse}')
print(f'MAE: {mae}')
.2
# Οπτικοποίηση των προβλέψεων
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, predictions)
# plt.xlabel('Πραγματικές Τιμές')
# plt.ylabel('Προβλέψεις')
# plt.title('Πραγματικές έναντι Προβλεπόμενων Τιμών')
# plt.show()


plt.scatter(X_test, y_test, color='black', label='Πραγματικές τιμές')
plt.scatter(X_test, predictions, color='blue', label='Προβλέψεις')
plt.xlabel('Τιμή Κλεισίματος')
plt.ylabel('Επόμενη Τιμή Κλεισίματος')
plt.title('Πραγματικές τιμές vs Προβλέψεις')
plt.legend()
plt.show()