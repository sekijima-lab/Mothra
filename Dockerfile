FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
RUN apt update
RUN apt install python3.9 python3.9-venv openbabel autodock-vina -y
WORKDIR /app

COPY requirements.txt /app
RUN cd /app
RUN python3.9 -m venv .venv
ENV PATH /app/.venv/bin:$PATH

#RUN source .venv/bin/activate
RUN pip install --upgrade pip

RUN pip install -r requirements.txt
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /mnt
CMD ["python", "--version"]