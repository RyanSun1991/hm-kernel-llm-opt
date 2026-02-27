#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ -n "${HMOPT_INSTALL_EDITABLE:-}" ]]; then
  python3 -m pip install --no-cache-dir -e .
fi

if [[ -n "${HMOPT_PATH_ALIAS:-}" ]]; then
  IFS=',' read -r -a _path_alias_pairs <<< "$HMOPT_PATH_ALIAS"
  for _pair in "${_path_alias_pairs[@]}"; do
    _from="${_pair%%:*}"
    _to="${_pair#*:}"
    [[ -z "${_from:-}" || -z "${_to:-}" || "$_from" == "$_pair" ]] && continue

    mkdir -p "$(dirname "$_from")"
    if [[ -L "$_from" || ! -e "$_from" ]]; then
      ln -sfn "$_to" "$_from"
      echo "[entrypoint] path alias: $_from -> $_to"
    else
      echo "[entrypoint] path alias skipped (exists and not symlink): $_from"
    fi
  done
fi

if [[ "${HMOPT_START_NEO4J:-1}" == "1" ]]; then
  NEO4J_USER="${NEO4J_USER:-neo4j}"
  NEO4J_PASSWORD="${NEO4J_PASSWORD:-@huawei2026}"

  mkdir -p /var/lib/neo4j/plugins
  APOC_JAR=""
  if compgen -G "/app/libs/apoc-*-core.jar" >/dev/null; then
    APOC_JAR="$(ls -1 /app/libs/apoc-*-core.jar | sort -V | tail -n1)"
  elif compgen -G "/app/libs/apoc-*.jar" >/dev/null; then
    APOC_JAR="$(ls -1 /app/libs/apoc-*.jar | sort -V | tail -n1)"
  elif compgen -G "/opt/hmopt-libs/apoc-*-core.jar" >/dev/null; then
    APOC_JAR="$(ls -1 /opt/hmopt-libs/apoc-*-core.jar | sort -V | tail -n1)"
  elif compgen -G "/opt/hmopt-libs/apoc-*.jar" >/dev/null; then
    APOC_JAR="$(ls -1 /opt/hmopt-libs/apoc-*.jar | sort -V | tail -n1)"
  fi

  if [[ -n "$APOC_JAR" ]]; then
    cp -f "$APOC_JAR" /var/lib/neo4j/plugins/apoc.jar
    chown neo4j:neo4j /var/lib/neo4j/plugins/apoc.jar || true
    echo "[entrypoint] APOC plugin installed from $APOC_JAR"
  else
    echo "[entrypoint] APOC plugin not found in /app/libs or /opt/hmopt-libs (skip)"
  fi

  if [[ ! -f /var/lib/neo4j/.hmopt_password_initialized ]]; then
    neo4j-admin dbms set-initial-password "$NEO4J_PASSWORD" || true
    touch /var/lib/neo4j/.hmopt_password_initialized
  fi

  neo4j console >/tmp/neo4j.log 2>&1 &
fi

exec "$@"
