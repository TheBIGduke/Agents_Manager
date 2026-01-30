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

sync_repo() {
  local REPO_URL="${1}"
  local DEST_DIR="${2}"

  # Require both args
  if [[ -z "${REPO_URL}" || -z "${DEST_DIR}" ]]; then
    err "Usage: sync_local_llm_repo <REPO_URL> <DEST_DIR>"
    return 2
  fi

  log "Syncing local_agent_module repo into ${DEST_DIR}…"
  log "Repo: ${REPO_URL}"

  if [[ -d "${DEST_DIR}/.git" ]]; then
    # Existing git repo: pull latest
    ( cd "${DEST_DIR}" && git pull --rebase )
    ok "local_agent_module updated via git pull."
    return 0
  fi

  if [[ -d "${DEST_DIR}" && ! -d "${DEST_DIR}/.git" ]]; then
    warn "${DEST_DIR} exists but is not a git repository. Skipping clone to avoid overwriting."
    warn "If you want a fresh clone, rename or remove that folder and re-run the installer."
    return 0
  fi

  # Folder does not exist: clone
  git clone "${REPO_URL}" "${DEST_DIR}"
  ok "local_agent_module cloned into ${DEST_DIR}."
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

create_cache_directory(){
  # Create RAG_documents directory if it doesn't exist (ignore if it does)
  mkdir -p "$HOME/.agents_manager/online_agent/RAG_documents"

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
  
  3) Remember to add your RAG for online_agent_module
     in PATH: ~/.agents_manager/online_agent/RAG_documents

Tip: The cache lives in ~/.agent_manager
────────────────────────────────────────────────────────────
EOS
}

main() {
  require_sudo
  install_apt
  sync_repo "https://github.com/JossueE/Local-LLM.git" "$HOME/local_agent_module"
  sync_repo "https://springlabsdevs.net/mecatronica/digital-signage/agent_core.git" "$HOME/online_agent_module"
  # Note: We no longer need ensure_snapd or install_yq 
  # because we use python for downloading models.
  create_venv_and_install
  download_models
  create_cache_directory
  post_instructions
}

main "$@"