FROM python:3.8

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install http://download.pytorch.org/whl/cpu/torch-1.7.1%2Bcpu-cp38-cp38m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cpu/torchvision-0.8.2%2Bcpu-cp38-cp38m-linux_x86_64.whl
RUN pip --no-cache-dir install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]
CMD ["bird_classification_website.py"]