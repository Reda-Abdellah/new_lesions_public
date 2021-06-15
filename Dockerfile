FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils ca-certificates wget unzip git git-lfs python3 python3-pip python-is-python3 pip
RUN pip install nibabel torch statsmodels
RUN update-ca-certificates
WORKDIR /anima/
RUN wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.0.1/Anima-Ubuntu-4.0.1.zip
RUN unzip Anima-Ubuntu-4.0.1.zip
RUN git lfs install
#RUN git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Public.git
COPY Anima-Scripts-Public /anima/Anima-Scripts-Public
RUN git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
RUN mkdir /root/.anima/
COPY config.txt /root/.anima
RUN mkdir /anima/WEIGHTS
#COPY 2times_in_iqda_v2/* /anima/WEIGHTS
COPY 2times_in_iqda_v2.zip /anima/WEIGHTS
RUN unzip 2times_in_iqda_v2.zip
RUN cp 2times_in_iqda_v2/* ../
COPY *.py /anima/
RUN mkdir -p /data/patients/patient_X/
COPY /data/patients/patient_X/* /data/patients/patient_X/

ENTRYPOINT ["python3", "process.py"]
