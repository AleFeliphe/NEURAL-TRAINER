import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from urllib.parse import urlparse
import socket

# Função para analisar a URL e extrair as características
class URLAnalyzer:
    def __init__(self, url):
        self.url = url
        self.parsed_url = urlparse(url)

    def is_https(self):
        return int(self.parsed_url.scheme == 'https')

    def count_dots(self):
        return self.url.count('.')

    def url_length(self):
        return len(self.url)

    def count_digits(self):
        return sum(c.isdigit() for c in self.url)

    def count_special_characters(self):
        special_characters = ":;#!%~+_?=&[]"
        return sum(1 for c in self.url if c in special_characters)

    def count_hyphens(self):
        return self.url.count('-')

    def count_double_slashes(self):
        return self.url.count('//')

    def count_slashes(self):
        return self.url.count('/') - self.count_double_slashes()

    def count_at_symbols(self):
        return self.url.count('@')

    def is_ip_address(self):
        hostname = self.parsed_url.hostname
        if not hostname:
            return False

        try:
            socket.inet_pton(socket.AF_INET, hostname)
            return True
        except socket.error:
            try:
                socket.inet_pton(socket.AF_INET6, hostname)
                return True
            except socket.error:
                return False

    def extract_features(self):
        features = [
            self.is_https(),
            self.count_dots(),
            self.url_length(),
            self.count_digits(),
            self.count_special_characters(),
            self.count_hyphens(),
            self.count_double_slashes(),
            self.count_slashes(),
            self.count_at_symbols(),
            self.is_ip_address()
        ]
        return features

# Função para carregar URLs e rotulá-las (1 para phishing, 0 para não phishing)
def load_urls_and_labels(phishing_file, non_phishing_file):
    phishing_urls = []
    non_phishing_urls = []
    
    # Carregar links de phishing
    with open(phishing_file, 'r') as f:
        phishing_urls = f.readlines()
        
    # Carregar links de não phishing
    with open(non_phishing_file, 'r') as f:
        non_phishing_urls = f.readlines()
    
    # Remover quebras de linha e criar rótulos
    phishing_urls = [url.strip() for url in phishing_urls]
    non_phishing_urls = [url.strip() for url in non_phishing_urls]

    # Criar os rótulos (1 para phishing, 0 para não phishing)
    urls = phishing_urls + non_phishing_urls
    labels = [1] * len(phishing_urls) + [0] * len(non_phishing_urls)
    
    return urls, labels

# Carregar URLs e rótulos
phishing_file = "phishing-links-ACTIVE-today.txt"
non_phishing_file = "trueurllist.txt"

urls, labels = load_urls_and_labels(phishing_file, non_phishing_file)

# Extrair características de cada URL
analyzers = [URLAnalyzer(url) for url in urls]
features = [analyzer.extract_features() for analyzer in analyzers]

# Converter para array numpy para treino
X = np.array(features)
y = np.array(labels)

# Dividir dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalonamento das características (normalização)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir o modelo neural com Dropout
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),  # Camada Dropout para evitar overfitting
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Camada de saída com sigmoide para probabilidade de phishing
])

# Compilar o modelo
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback de EarlyStopping para evitar overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Função para ajuste da taxa de aprendizado (Learning Rate Scheduler)
def lr_scheduler(epoch):
    initial_lr = 0.001
    drop_factor = 0.5
    epoch_drop = 10
    return initial_lr * (drop_factor ** (epoch // epoch_drop))

# Callback de LearningRateScheduler
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# Treinar o modelo
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), 
          callbacks=[early_stopping, lr_scheduler_callback])

# Salvar o modelo treinado e o scaler
model.save('url_classifier_model.h5')
joblib.dump(scaler, 'scaler.pkl')

print("Modelo e scaler salvos com sucesso!")

