FROM amd64/ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
WORKDIR /workdir
RUN cd /workdir
RUN apt update
RUN apt upgrade -y
RUN apt install wget cmake gcc g++ libboost-dev git tzdata software-properties-common zip -y
RUN apt autoremove -y
#CUDA install
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt update
RUN apt install cuda -y
# get Autodock Vina 1.1.2(Latest: 1.2.0 but required source build)
RUN wget "https://vina.scripps.edu/wp-content/uploads/sites/55/2020/12/autodock_vina_1_1_2_linux_x86.tgz"
RUN tar xvzf autodock_vina_1_1_2_linux_x86.tgz
RUN rm autodock_vina_1_1_2_linux_x86.tgz
ENV PATH=$PATH:/workdir/autodock_vina_1_1_2_linux_x86/bin
# get Open Babel 3.1.1
RUN wget "https://github.com/openbabel/openbabel/releases/download/openbabel-3-1-1/openbabel-3.1.1-source.tar.bz2"
RUN tar xf openbabel-3.1.1-source.tar.bz2
RUN rm openbabel-3.1.1-source.tar.bz2
RUN cd openbabel-3.1.1
WORKDIR /workdir/openbabel-3.1.1/build
RUN cmake ..
RUN make -j6
RUN make install
# get python
#RUN add-apt-repository ppa:deadsnakes/ppa
#RUN apt update
#RUN apt full-upgrade -y
#RUN apt autoremove -y
RUN apt install python3 python3-pip -y
RUN ln -s /usr/bin/python3 /usr/bin/python
#RUN ln -s /usr/lib/python3/dist-packages/pip /usr/bin/pip
RUN python -m pip install --upgrade pip
RUN python -m pip install black jupyterlab jupyterlab_code_formatter jupyterlab-git lckr-jupyterlab-variableinspector jupyterlab_widgets ipywidgets import-ipynb
RUN python -m pip install matplotlib ipymol ipympl pandas numpy rdkit
# get pymol
#WORKDIR /workdir
#RUN wget https://pymol.org/installers/PyMOL-2.5.4_404-Linux-x86_64-py37.tar.bz2
#RUN tar -jxf PyMOL-2.5.4_404-Linux-x86_64-py37.tar.bz2
#ENV PATH=$PATH:/workdir/pymol
RUN apt install pymol -y
WORKDIR /workspace
CMD [ "python","--version",";" ,"pymol","--version"]