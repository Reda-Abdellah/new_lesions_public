FROM ubuntu:20.04
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils ca-certificates wget unzip git git-lfs python3 python3-pip python-is-python3 pip
RUN pip install nibabel torch statsmodels
RUN update-ca-certificates
WORKDIR /anima/
RUN wget -q https://github.com/Inria-Visages/Anima-Public/releases/download/v4.0.1/Anima-Ubuntu-4.0.1.zip
RUN unzip Anima-Ubuntu-4.0.1.zip
RUN git lfs install
RUN git clone --depth 1 https://github.com/Inria-Visages/Anima-Scripts-Data-Public.git
RUN mkdir /root/.anima/
RUN mkdir /anima/WEIGHTS
RUN mkdir /tmp/test_image
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NSZilDWXLcHAHP0FyGs945hjj62ypvWy' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NSZilDWXLcHAHP0FyGs945hjj62ypvWy" -O 2times_in_iqda_v2.zip && rm -rf /tmp/cookies.txt
RUN unzip 2times_in_iqda_v2.zip
RUN git clone https://github.com/Reda-Abdellah/new_lesions_public.git
RUN cp -avr new_lesions_public/Anima-Scripts-Public /anima/Anima-Scripts-Public
RUN cp new_lesions_public/config.txt /root/.anima
RUN cp 2times_in_iqda_v2/* /anima/WEIGHTS
RUN cp new_lesions_public/*.py /anima/
RUN mkdir -p /data/patients/patient_X/


###To Run just do:
### sudo docker build -t new_lesions_inf .
### sudo docker run -v [FOLDER Containing input and output]:/tmp/ new_lesions_inf -t1 [PATH_time1] -t2 [PATH_time2] -o [PATH_output]
