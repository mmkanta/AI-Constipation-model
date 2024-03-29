FROM python:3.8

WORKDIR /code
RUN apt-get update && apt-get install libgl1 -y
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

EXPOSE 7000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--workers", "3"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7000", "--reload"]