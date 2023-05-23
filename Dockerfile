FROM ubuntu

RUN apt-get update && apt-get install -y python3 python3-pip unzip

RUN python3 -m pip install pandas numpy tensorflow imbalanced-learn sklearn sacred pymongo mlflow
RUN apt-get install -y git

COPY train.py /app/train.py
COPY predictions.py /app/predictions.py
COPY data.csv /app/data.csv

WORKDIR /app

RUN export SACRED_IGNORE_GIT=TRUE
RUN python3 train.py --epochs 10

CMD ["python3", "predictions.py"]
