FROM nvidia/cuda:12.5.1-devel-ubuntu22.04

ENV USERNAME=dockeruser

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,display,video
ENV DEBIAN_FRONTEND noninteractive

# Combine installation commands to reduce layers
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    sudo ssh tmux git cmake htop vim apt-utils wget build-essential curl dirmngr pkg-config gpg-agent libdrm-dev \
    libglvnd0 libglx0 libxext6 libx11-6 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

   
# Miniconda installation
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/opt/conda/bin:$PATH
RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && echo ". /opt/conda/etc/profile.d/conda.sh" >> /.bashrc

# Create and activate conda environment
COPY requirements.txt requirements.txt
RUN conda init bash && \
    conda create --name steering python=3.10 --channel conda-forge -y && \
    echo "conda activate steering" >> /.bashrc && \
    /bin/bash -c "source /.bashrc && conda activate steering && conda install pip -y && pip install -r requirements.txt"

# Cleanup
RUN apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy 10_nvidia.json file
COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Expose necessary ports
EXPOSE 22 42421 6006 8888

# User management and directory setup
RUN useradd -m -s /bin/bash $USERNAME && \
    usermod -aG sudo,video,audio $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

WORKDIR /home/$USERNAME
RUN mkdir /repositories && chown -R $USERNAME:$USERNAME /repositories
WORKDIR /repositories

# Switch to non-root user
USER $USERNAME

ENV DEBIAN_FRONTEND teletype

# /.bashrc should always be sourced 
CMD ["/bin/bash", "--rcfile", "/.bashrc"]