#dockerfile

# Versione di Python
FROM python:3.12.1

# Set della working directory  
WORKDIR /app
RUN ls

# Copia il file dei requisiti
COPY requirements.txt .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia della directory in /app
COPY . /app

# Run dello script Python
CMD ["python", "app.py"]