FROM python:3.8

COPY . /app

WORKDIR /app/src

#We need to install torch CPU version because we're just serving the application and the CPU version is much smaller
#Installing it separately because we need to install it from a specific location

#RUN pip --no-cache-dir install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

#The rest of the necessary packages can be installed from requirements.txt with no cache dir allowing for installation on
#machines with very little memory on board

RUN pip --no-cache-dir install -r ../requirements.txt

#Exposing the default streamlit port
EXPOSE 8501

#Running the streamlit app
ENTRYPOINT ["streamlit", "run"]
CMD ["bird_classification_website.py"]