ARG BASE_IMAGE=YOUR_REGISTRY/hmci-docker-image:v3-4.2
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=pypi.org
ARG NEO4J_REPO_URL=https://debian.neo4j.com
ARG NEO4J_VERIFY_PEER=false
ARG NEO4J_INSTALL_MODE=offline

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    wget \
    git \
    gnupg \
    ca-certificates \
    lsb-release \
    openjdk-17-jre-headless \
    clangd \
    && rm -rf /var/lib/apt/lists/*

COPY docker/neo4j-offline /tmp/neo4j-offline

# Install Neo4j in image (single-container mode).
# - offline mode: use host-downloaded neo4j/cypher-shell .deb from build context
# - online mode: fetch from debian.neo4j.com during docker build
RUN if [ "$NEO4J_INSTALL_MODE" = "offline" ] && ls /tmp/neo4j-offline/neo4j*.deb >/dev/null 2>&1; then \
      apt-get update; \
      apt-get install -y --no-install-recommends /tmp/neo4j-offline/*.deb; \
    else \
      mkdir -p /etc/apt/keyrings; \
      curl --retry 5 --retry-delay 5 --connect-timeout 15 -fsSL ${NEO4J_REPO_URL}/neotechnology.gpg.key | gpg --dearmor -o /etc/apt/keyrings/neo4j.gpg; \
      echo "deb [signed-by=/etc/apt/keyrings/neo4j.gpg] ${NEO4J_REPO_URL} stable latest" > /etc/apt/sources.list.d/neo4j.list; \
      printf 'Acquire::https::debian.neo4j.com::Verify-Peer "%s";\n' "${NEO4J_VERIFY_PEER}" > /etc/apt/apt.conf.d/99neo4j-insecure; \
      apt-get update && apt-get install -y --no-install-recommends neo4j; \
    fi \
    && rm -rf /var/lib/apt/lists/* /tmp/neo4j-offline

RUN if [ -f /etc/neo4j/neo4j.conf ]; then \
      sed -i 's|#dbms.security.procedures.unrestricted=.*|dbms.security.procedures.unrestricted=apoc.*|' /etc/neo4j/neo4j.conf; \
      sed -i 's|#dbms.security.procedures.allowlist=.*|dbms.security.procedures.allowlist=apoc.*|' /etc/neo4j/neo4j.conf; \
      echo 'dbms.security.auth_enabled=true' >> /etc/neo4j/neo4j.conf; \
      echo 'server.default_listen_address=0.0.0.0' >> /etc/neo4j/neo4j.conf; \
      echo 'server.http.listen_address=:7474' >> /etc/neo4j/neo4j.conf; \
      echo 'server.bolt.listen_address=:7687' >> /etc/neo4j/neo4j.conf; \
    fi

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY examples ./examples
COPY libs ./libs
COPY libs /opt/hmopt-libs

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    -i "${PIP_INDEX_URL}" && \
    python3 -m pip install --no-cache-dir -e . \
    --trusted-host "${PIP_TRUSTED_HOST}" \
    -i "${PIP_INDEX_URL}" && \
    if ls libs/*.whl >/dev/null 2>&1; then \
      python3 -m pip install --no-cache-dir --trusted-host "${PIP_TRUSTED_HOST}" -i "${PIP_INDEX_URL}" libs/*.whl; \
    fi

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 7474 7687 7331 8000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-lc", "tail -f /dev/null"]
