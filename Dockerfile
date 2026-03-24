#dockerfile

# versione di Python
FROM python:3.12.1

# copia della directory in /app
COPY ./MLOps_project /app

# set della working directory  
WORKDIR /app
RUN ls

# run dello script Python
CMD ["python", "app.py"]