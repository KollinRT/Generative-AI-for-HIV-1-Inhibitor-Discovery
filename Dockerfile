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
