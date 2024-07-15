FROM python
WORKDIR mlapp
COPY rental.py rental.py
COPY london_processed_data.csv data/london_processed_data.csv
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python", "rental.py"]
EXPOSE 5000
