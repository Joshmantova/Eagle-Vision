FROM python:3.7

COPY ./requirements.txt /app/requirements.txt
COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

CMD streamlit run bird_classification_website.py