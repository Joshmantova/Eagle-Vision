FROM python:3.8

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["bird_classification_website.py"]