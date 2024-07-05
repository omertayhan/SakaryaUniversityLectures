import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Veri setini pandas DataFrame'e yükleme
df = pd.read_csv('data/hepatitis.csv')

# Eksik değerleri her sütunun ortalaması ile doldurma
for col in df.columns:
    if df[col].dtype != 'object':  # Sadece sayısal sütunları kontrol et
        df[col].fillna(df[col].mean(), inplace=True)

# Özellikler ve hedef değişken olarak ayırma
X = df.drop('class', axis=1)  # Özellikler
y = df['class']  # Hedef değişken

# Kategorik verileri sayısal değerlere dönüştürme (gerekiyorsa)
X = pd.get_dummies(X)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes sınıflandırıcıyı oluşturma ve eğitme
model = GaussianNB()
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Doğruluk değerini hesaplama
accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy}")
