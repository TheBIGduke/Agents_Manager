#!/usr/bin/env bash

# installer.sh — One-shot installer
# Tested on Ubuntu/Debian-like systems.

set -euo pipefail

APT_PKGS=(
  python3-dev python3-venv build-essential curl unzip
  portaudio19-dev ffmpeg
)

log()   { printf "\n\033[1;34m[INFO]\033[0m %s\n" "$*"; }
ok()    { printf "\033[1;32m[OK]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
fail()  { printf "\033[1;31m[FAIL]\033[0m %s\n" "$*"; exit 1; }

require_sudo() {
  if [[ $EUID -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      SUDO="sudo"
    else
      fail "Run as root or install sudo."
    fi
  else
    SUDO=""
  fi
}

install_apt() {
  log "Updating apt and installing base packages…"
  $SUDO apt update -y
  $SUDO apt install -y "${APT_PKGS[@]}"
  ok "System packages installed."
}

create_venv_and_install() {
  log "Creating Python virtual environment…"
  python3 -m venv ".venv" || fail "venv creation failed."
  ok "venv created: .venv"

  log "Installing Python dependencies into the venv…"
  ".venv/bin/pip" install --upgrade pip
  ".venv/bin/pip" install -r "requirements.txt"
  ok "Dependencies installed."
}

download_models() {
  log "Downloading/verifying models using Python..."
  # We use the python binary from the venv we just created
  if [[ -f ".venv/bin/python" ]]; then
      ".venv/bin/python" utils/download.py
  else
      fail "Virtual environment python not found. Install failed."
  fi
  ok "Model download/verify step completed."
}

post_instructions() {
  cat <<'EOS'

────────────────────────────────────────────────────────────
Setup completed!

Next steps:
  1) Activate the virtual environment:
       source .venv/bin/activate

  2) Run the assistant:
       python -m main

Tip: The cache lives in ~/.cache/agent_manager
────────────────────────────────────────────────────────────
EOS
}

main() {
  require_sudo
  install_apt
  # Note: We no longer need ensure_snapd or install_yq 
  # because we use python for downloading models.
  create_venv_and_install
  download_models
  post_instructions
}

main "$@"