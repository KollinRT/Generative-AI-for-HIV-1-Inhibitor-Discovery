# FROM ubuntu:latest

# RUN apt-get update \
#     && apt-get install -y \
#         nmap \
#         vim \
#         python3 \
#         python3-pip \
#         gcc \ 
#         mysql-client \
#         mysql-server \
#         wget

# # The installer requires curl (and certificates) to download the release archive
# RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# # Download the latest installer
# ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # Run the installer then remove it
# RUN sh /uv-installer.sh && rm /uv-installer.sh

# # Ensure the installed binary is on the `PATH`
# ENV PATH="/root/.local/bin/:$PATH"

# # Ensure the data directory exists
# RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

# # Expose MySQL port
# EXPOSE 3306

# # Initialize MySQL if needed and start server
# CMD ["bash", "-c", "mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql; exec mysqld_safe --datadir=/var/lib/mysql"]


# # Build
# # docker compose build
# # run
# # docker compose up -d(etached)
# # Breakdown 
# # docker compose down -v
# # Build

# # Get into docker file container
# # docker exec -it mysql-container-test bash

FROM ubuntu:latest

# Install dependencies
RUN apt-get update && apt-get install -y \
    nmap \
    vim \
    wget \
    curl \
    ca-certificates \
    bzip2 \
    gcc \
    mysql-client \
    mysql-server \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda clean -afy

# Need to get the documents...
COPY working_updated_environment.yml ./


RUN conda tos view
RUN conda tos accept

# # Install dependencies
# RUN conda env create -f working_updated_environment.yml

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
## Install env via UV
RUN uv sync
#

# Ensure UV binary path is on PATH (if different from ~/.local/bin)
ENV PATH="/root/.local/bin:$PATH"

# Ensure MySQL data directory exists
RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

# Expose MySQL port
EXPOSE 3306

# Initialize MySQL if needed and start server
CMD ["bash", "-c", "mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql; exec mysqld_safe --datadir=/var/lib/mysql"]
