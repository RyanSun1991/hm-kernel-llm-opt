#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-docker/neo4j-offline}"
NEO4J_KEY_URL="${NEO4J_KEY_URL:-https://debian.neo4j.com/neotechnology.gpg.key}"
NEO4J_APT_URL="${NEO4J_APT_URL:-https://debian.neo4j.com}"

mkdir -p "$OUT_DIR"

echo "[prepare-neo4j-offline] downloading Neo4j artifacts on host..."

sudo mkdir -p /etc/apt/keyrings

if curl -fsSL "$NEO4J_KEY_URL" | sudo gpg --dearmor --batch --yes -o /etc/apt/keyrings/neo4j.gpg; then
  echo "[prepare-neo4j-offline] downloaded apt key with TLS verification"
else
  echo "[prepare-neo4j-offline] WARN: key download failed with TLS verification, retrying with curl -k"
  curl -kfsSL "$NEO4J_KEY_URL" | sudo gpg --dearmor --batch --yes -o /etc/apt/keyrings/neo4j.gpg
fi

echo "deb [signed-by=/etc/apt/keyrings/neo4j.gpg] ${NEO4J_APT_URL} stable latest" | sudo tee /etc/apt/sources.list.d/neo4j.list >/dev/null
echo 'Acquire::https::debian.neo4j.com::Verify-Peer "false";' | sudo tee /etc/apt/apt.conf.d/99neo4j-insecure >/dev/null

sudo apt-get update

(
  cd "$OUT_DIR"
  rm -f neo4j*.deb cypher-shell*.deb
  apt-get download neo4j cypher-shell
)

cp /etc/apt/keyrings/neo4j.gpg "$OUT_DIR/neo4j.gpg"

echo "[prepare-neo4j-offline] done. Files in: $OUT_DIR"
ls -lh "$OUT_DIR"
