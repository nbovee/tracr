FROM python:3.11.7

WORKDIR /usr/src/tracr/

COPY ./requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 9000
