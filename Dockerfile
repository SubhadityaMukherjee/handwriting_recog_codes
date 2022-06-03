# Our base image
FROM tensorflow/tensorflow:2.2.3-gpu
 
# Copy the requirements.txt file to our Docker image
ADD requirements.txt .
 
# Install the requirements.txt
RUN pip install -r requirements.txt
RUN pip install pandas IPython python-Levenshtein
 
# Some common environmenta variables that Python uses
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list


# Install lower level dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y curl python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*
 
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
 
ADD . /hwr