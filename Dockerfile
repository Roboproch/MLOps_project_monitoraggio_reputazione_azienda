#dockerfile

# versione di Python
FROM python:3.12.1

# set della working directory  
WORKDIR /app
RUN ls

# Copia il file dei requisiti
COPY requirements.txt .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# copia della directory in /app
COPY . /app

# run dello script Python
CMD ["python", "app.py"]