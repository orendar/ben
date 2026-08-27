#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# BEN native Linux setup (Ubuntu 24.04, including WSL "Ubuntu-24.04").
#
# Standardises on Python 3.12 in a venv and installs everything the engine
# needs at runtime: system packages, Python requirements, and the .NET runtime
# used by the ACE / BBA engines. Idempotent - safe to re-run.
#
# Usage (from anywhere inside the repo):
#     bash setup_linux.sh
#
# Prerequisite: the distro itself. On Windows, once:  wsl --install Ubuntu-24.04
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${BEN_VENV:-$HOME/ben-venv}"
DOTNET_DIR="${DOTNET_ROOT:-$HOME/.dotnet}"

echo "[ben-setup] repo:  $REPO_DIR"
echo "[ben-setup] venv:  $VENV_DIR"

if [ "$VENV_DIR" = "$REPO_DIR" ]; then
    echo "[ben-setup] BEN_VENV must not be the repo itself ($REPO_DIR)." >&2
    exit 1
fi

# The vendored bin/dds3-linux extension needs glibc >= 2.38 and Python 3.12, and
# libicu is versioned per release: 24.04 ships 74, 25.x/26.x ship 78. Asking for
# the wrong one is a hard apt failure, so pick it from the box.
. /etc/os-release 2>/dev/null || true
case "${VERSION_ID:-}" in
    24.04)     ICU_PKG=libicu74 ;;
    25.*|26.*) ICU_PKG=libicu78 ;;
    *)
        ICU_PKG=libicu78
        echo "[ben-setup] WARNING: unrecognised '${PRETTY_NAME:-unknown}'; guessing $ICU_PKG."
        echo "            See src/ddsolver/README.md if the dds3 import fails." ;;
esac

# Containers run as root and usually ship no sudo.
SUDO=sudo
[ "$(id -u)" -eq 0 ] && SUDO=""

# 1. System packages
#    python3-gdbm: gives shelve a real dbm backend (without it shelve uses
#    dbm.dumb, which chmods its files and fails on filesystems such as WSL /mnt).
echo "[ben-setup] installing system packages ($ICU_PKG) ..."
$SUDO apt-get update
$SUDO apt-get install -y \
    python3.12 python3.12-venv python3-pip python3-gdbm \
    "$ICU_PKG" curl libboost-thread-dev

# 2. Python venv + requirements (venv inherits the system stdlib, incl. _gdbm)
if [ ! -d "$VENV_DIR" ]; then
    echo "[ben-setup] creating venv at $VENV_DIR ..."
    python3.12 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "[ben-setup] installing Python requirements ..."
python -m pip install --upgrade pip
python -m pip install -r "$REPO_DIR/requirements.txt"

# 3. .NET 10 runtime for the ACE / BBA (.NET) engines
if [ ! -x "$DOTNET_DIR/dotnet" ]; then
    echo "[ben-setup] installing .NET 10 runtime to $DOTNET_DIR ..."
    curl -sSL https://dot.net/v1/dotnet-install.sh \
        | bash /dev/stdin --channel 10.0 --runtime dotnet --install-dir "$DOTNET_DIR"
else
    echo "[ben-setup] .NET runtime already present at $DOTNET_DIR"
fi

# 4. Persist DOTNET_ROOT and a convenience alias in ~/.bashrc (guarded)
BRC="$HOME/.bashrc"
if ! grep -q '# >>> BEN setup >>>' "$BRC" 2>/dev/null; then
    echo "[ben-setup] adding env to $BRC ..."
    cat >> "$BRC" <<EOF

# >>> BEN setup >>>
export DOTNET_ROOT="$DOTNET_DIR"
export PATH="\$DOTNET_ROOT:\$PATH"
alias ben-activate='source "$VENV_DIR/bin/activate"'
# <<< BEN setup <<<
EOF
fi
export DOTNET_ROOT="$DOTNET_DIR"
export PATH="$DOTNET_DIR:$PATH"

echo
echo "[ben-setup] done."
echo "  Activate:  source $VENV_DIR/bin/activate     (or 'ben-activate' in a new shell)"
echo "  Test:      cd $REPO_DIR/src && python game.py --boards ../Challenges/martens_declarer_first10.pbn --auto true"
