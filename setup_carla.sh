#!/usr/bin/env bash

set -e

CARLA_URL="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.10.1.tar.gz"
MAPS_URL="https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.10.1.tar.gz"

download() {
  local url="$1"
  local out="$2"

  # Prefer aria2c for faster parallel downloads if available.
  if command -v aria2c >/dev/null 2>&1; then
    echo "Using aria2c for fast download of ${out}..."
    aria2c -c -x 8 -s 8 -o "${out}" "${url}"
  else
    echo "aria2c not found; falling back to wget. Install aria2c for faster downloads."
    wget -c -O "${out}" "${url}"
  fi
}

# Download and install CARLA
mkdir -p carla
cd carla

download "${CARLA_URL}" "CARLA_0.9.10.1.tar.gz"
download "${MAPS_URL}" "AdditionalMaps_0.9.10.1.tar.gz"

tar -xf CARLA_0.9.10.1.tar.gz
tar -xf AdditionalMaps_0.9.10.1.tar.gz
rm CARLA_0.9.10.1.tar.gz
rm AdditionalMaps_0.9.10.1.tar.gz
cd ..
