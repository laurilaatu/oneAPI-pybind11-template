#Build:
#docker build -t oneapi-2025 --build-arg user=$USER .
#Run:
#docker run -p 127.0.0.1:8080:8080 -v /home/$USER/:/home/$USER/local -it oneapi-2025

# Run with GPU support:
# docker run -it \
#  --device /dev/dri \
#  --security-opt label=disable \
#  -v /home/$USER/:/home/$USER/local \
#  -p 8080:8080 \
#  oneapi-2025

# Inside the container:
# source /opt/intel/oneapi/2025.3/oneapi-vars.sh --force
# jupyter notebook --port=8080 --ip=0.0.0.0 --no-browser


FROM intel/oneapi-basekit:2025.3.1-0-devel-ubuntu24.04

ARG user=jupyter

SHELL ["/bin/bash", "-c"]

# Install wget to fetch Mini-forge and other apt dependencies
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean

# Ensure groups exist, delete ubuntu user, and create the new user in the video and render groups
RUN userdel ubuntu || true
RUN groupadd -f render && \
    groupadd -f video && \
    useradd -m $user -s /bin/bash -G video,render

USER $user

ENV PATH="/home/$user/miniconda3/bin:$PATH"

RUN MINICONDA_URL="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"; \
    cd ;\
    pwd ; \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /home/$user/.conda && \
    bash miniconda.sh -b -p /home/$user/miniconda3 && \
    rm -f miniconda.sh


# Change SHELL so that RUN commands are run inside conda base env
SHELL ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]


# Create conda envs and install packages
RUN conda create -n oneapi-env python=3.14 -y \
    && conda run -n oneapi-env pip install matplotlib scikit-learn ndjson pyparsing \
    && conda run -n oneapi-env conda install -y jupyter \
    && conda run -n oneapi-env conda install -y -c conda-forge libstdcxx-ng pybind11 \
    && conda init bash

RUN echo "conda activate oneapi-env" >> ~/.bashrc

CMD ["/bin/bash"]

EXPOSE 8080