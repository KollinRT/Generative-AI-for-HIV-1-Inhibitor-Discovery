# # FROM ubuntu:latest

# # RUN apt-get update \
# #     && apt-get install -y \
# #         nmap \
# #         vim \
# #         python3 \
# #         python3-pip \
# #         gcc \ 
# #         mysql-client \
# #         mysql-server \
# #         wget

# # # The installer requires curl (and certificates) to download the release archive
# # RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# # # Download the latest installer
# # ADD https://astral.sh/uv/install.sh /uv-installer.sh

# # # Run the installer then remove it
# # RUN sh /uv-installer.sh && rm /uv-installer.sh

# # # Ensure the installed binary is on the `PATH`
# # ENV PATH="/root/.local/bin/:$PATH"

# # # Ensure the data directory exists
# # RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

# # # Expose MySQL port
# # EXPOSE 3306

# # # Initialize MySQL if needed and start server
# # CMD ["bash", "-c", "mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql; exec mysqld_safe --datadir=/var/lib/mysql"]


# # # Build
# # # docker compose build
# # # run
# # # docker compose up -d(etached)
# # # Breakdown 
# # # docker compose down -v
# # # Build

# # # Get into docker file container
# # # docker exec -it mysql-container-test bash

# # FROM ubuntu:latest

# # # Install dependencies
# # RUN apt-get update && apt-get install -y \
# #     nmap \
# #     vim \
# #     wget \
# #     curl \
# #     ca-certificates \
# #     bzip2 \
# #     gcc \
# #     mysql-client \
# #     mysql-server \
# #     && rm -rf /var/lib/apt/lists/*

# # # Set a working directory for your app / venv
# # WORKDIR /app


# # # # Install Miniconda
# # # ENV CONDA_DIR=/opt/conda
# # # ENV PATH=$CONDA_DIR/bin:$PATH

# # # RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
# # #     && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
# # #     && rm /tmp/miniconda.sh \
# # #     && conda clean -afy

# # # Need to get the documents...
# # # COPY working_updated_environment.yml ./
# # # COPY pyproject.toml ./
# # COPY pyproject.toml /app/


# # # RUN conda tos view
# # # RUN conda tos accept

# # # # Install dependencies
# # # RUN conda env create -f working_updated_environment.yml

# # # Install UV
# # # ADD https://astral.sh/uv/install.sh /uv-installer.sh
# # # RUN sh /uv-installer.sh && rm /uv-installer.sh
# # # ## Install env via UV
# # # RUN uv venv .venv
# # # RUN source .venv/bin/activate
# # # RUN uv sync
# # # #

# # ADD https://astral.sh/uv/install.sh /uv-installer.sh
# # RUN chmod +x /uv-installer.sh \
# #     && /uv-installer.sh \
# #     && rm /uv-installer.sh \
# #     && bash -lc '\
# #          uv venv .venv && \
# #          source .venv/bin/activate && \
# #          uv sync \
# #        '


# # # Ensure UV binary path is on PATH (if different from ~/.local/bin)
# # ENV PATH="/root/.local/bin:$PATH"

# # # Ensure MySQL data directory exists
# # RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

# # # Expose MySQL port
# # EXPOSE 3306

# # # Initialize MySQL if needed and start server
# # CMD ["bash", "-c", "mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql; exec mysqld_safe --datadir=/var/lib/mysql"]



# # NEWER

# # FROM ubuntu:latest

# # # Install dependencies
# # RUN apt-get update && apt-get install -y \
# #     python3 \
# #     python3-venv \
# #     python3-pip \
# #     python3-distutils \
# #     nmap \
# #     vim \
# #     wget \
# #     curl \
# #     ca-certificates \
# #     bzip2 \
# #     gcc \
# #     mysql-client \
# #     mysql-server \
# #     && rm -rf /var/lib/apt/lists/*

# # # Set a working directory for your app / venv
# # WORKDIR /app


# # # # Install Miniconda
# # # ENV CONDA_DIR=/opt/conda
# # # ENV PATH=$CONDA_DIR/bin:$PATH

# # # RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
# # #     && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
# # #     && rm /tmp/miniconda.sh \
# # #     && conda clean -afy

# # # Need to get the documents...
# # # COPY working_updated_environment.yml ./
# # # COPY pyproject.toml ./
# # COPY pyproject.toml /app/


# # # RUN conda tos view
# # # RUN conda tos accept

# # # # Install dependencies
# # # RUN conda env create -f working_updated_environment.yml

# # # Install UV
# # # ADD https://astral.sh/uv/install.sh /uv-installer.sh
# # # RUN sh /uv-installer.sh && rm /uv-installer.sh
# # # ## Install env via UV
# # # RUN uv venv .venv
# # # RUN source .venv/bin/activate
# # # RUN uv sync
# # # #

# # ADD https://astral.sh/uv/install.sh /uv-installer.sh
# # RUN chmod +x /uv-installer.sh \
# #     && /uv-installer.sh \
# #     && rm /uv-installer.sh \
# #     && bash -lc '\
# #          uv venv .venv && \
# #          source .venv/bin/activate && \
# #          uv sync \
# #        '


# # # Ensure UV binary path is on PATH (if different from ~/.local/bin)
# # ENV PATH="/root/.local/bin:$PATH"

# # # Ensure MySQL data directory exists
# # RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

# # # Expose MySQL port
# # EXPOSE 3306

# # # Initialize MySQL if needed and start server
# # CMD ["bash", "-c", "mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql; exec mysqld_safe --datadir=/var/lib/mysql"]

# FROM ubuntu:latest
# ENV DEBIAN_FRONTEND=noninteractive

# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-venv \
#     python3-pip \
#     python3-distutils \
#     wget curl ca-certificates bzip2 gcc \
#     mysql-client mysql-server \
#     && rm -rf /var/lib/apt/lists/*

# # Make `python` point to python3 for convenience
# RUN ln -s /usr/bin/python3 /usr/bin/python

# WORKDIR /app
# COPY pyproject.toml /app/

# # Install uv (it will use /usr/bin/python)
# ADD https://astral.sh/uv/install.sh /uv-installer.sh
# RUN chmod +x /uv-installer.sh \
#     && /uv-installer.sh && rm /uv-installer.sh \
#     && bash -lc '\
#          python -m venv .venv && \
#          source .venv/bin/activate && \
#          pip install --upgrade pip && \
#          uv venv .venv && \
#          uv sync \
#        '
# ENV PATH="/root/.local/bin:$PATH"
# # ...
# RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

# # Expose MySQL port
# EXPOSE 3306

# # Initialize MySQL if needed and start server
# CMD ["bash", "-c", "mysqld --initialize-insecure --user=mysql --datadir=/var/lib/mysql; exec mysqld_safe --datadir=/var/lib/mysql"]


FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive

# Ensure universe is available and basic package tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates \
    && add-apt-repository universe || true \
    && rm -rf /var/lib/apt/lists/*

# Install Python + tools and MySQL (try to install python3-distutils; if not available try python3.11-distutils)
RUN apt-get update && bash -lc '\
    apt-get install -y --no-install-recommends \
      python3 \
      python3-venv \
      python3-pip \
      python3-setuptools \
      vim \
      libxrender1 \
      libxext6 \
      libsm6 \
      wget \
      curl \
      bzip2 \
      gcc \
      mysql-client \
      mysql-server \
    && (apt-get install -y python3-distutils || apt-get install -y python3.11-distutils || true) \
    && rm -rf /var/lib/apt/lists/* \
'

# RUN apt-get update && apt-get install -y \
#     libxrender1 \
#     libxext6 \
#     libsm6

# Make `python` point to python3 for convenience
RUN ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY pyproject.toml /app/

# Install uv (will use the system python) and create venv + sync in same shell
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN chmod +x /uv-installer.sh \
    && /uv-installer.sh \
    && rm /uv-installer.sh \
    && bash -lc '\
         python -m venv .venv && \
         source .venv/bin/activate && \
         pip install --upgrade pip setuptools wheel || true && \
         uv venv .venv && \
         uv sync || true \
       '

ENV PATH="/root/.local/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Ensure MySQL data directory exists (ownership fixed at runtime if using volumes)
RUN mkdir -p /var/lib/mysql && chown -R mysql:mysql /var/lib/mysql

EXPOSE 3306

# # Default command: give a shell so you can interact. (You can override to start MySQL)
# CMD ["/bin/bash"]

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Use entrypoint to initialize MySQL and source venv
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD []
