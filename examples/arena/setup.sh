#!/usr/bin/env bash
# SO-101 example environment for Isaac Lab Arena, on the host (no Docker).
#
# Clones IsaacLab-Arena at ARENA_REV into ./IsaacLab-Arena, syncs Arena's own uv environment
# (Isaac Sim, Isaac Lab from Arena's submodule, Arena), installs arena-so101 from ../.. editable
# into it, and activates that environment in the current shell.
#
# Must be sourced so the activation sticks:
#   source ./setup.sh
#
# Options:
#   --force    Re-run uv sync and reinstall arena-so101 even if the environment exists
#   --leader   Install arena-so101 with its `leader` extra (lerobot, for a physical leader arm)
#   -h/--help  Show this help
#
# Env: ARENA_DIR — an existing IsaacLab-Arena checkout to use instead of cloning one. Its commit
# is then yours to choose; the example is tested at ARENA_REV.

_setup_main() {
  # IsaacLab-Arena commit this example is tested against (main, 2026-09-23). See README "Arena version".
  local ARENA_REV=aa36f191d89b1ea65fbba62b3350d6fc7d0e9b9f
  # A bare `local ARENA_DIR` would hide the caller's ARENA_DIR; copy its value in.
  local ARENA_DIR="${ARENA_DIR:-}"
  local HERE VENV MANAGED=true FORCE=false EXTRAS="" SYNCED=false
  HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  # A checkout the script made is kept at ARENA_REV; one you point ARENA_DIR at is left alone.
  if [[ -n "${ARENA_DIR}" ]]; then MANAGED=false; else ARENA_DIR="${HERE}/IsaacLab-Arena"; fi
  VENV="${ARENA_DIR}/.venv"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) FORCE=true ;;
      --leader) EXTRAS="[leader]" ;;
      -h|--help)
        sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'
        return 0
        ;;
      *)
        echo "setup.sh: unknown option: $1 (try --help)" >&2
        return 2
        ;;
    esac
    shift
  done

  if ! command -v uv >/dev/null 2>&1; then
    echo "setup.sh: uv not found on PATH. Install: https://docs.astral.sh/uv/" >&2
    return 1
  fi

  if [[ ! -f "${ARENA_DIR}/pyproject.toml" ]]; then
    echo "setup.sh: cloning IsaacLab-Arena @ ${ARENA_REV:0:9} into ${ARENA_DIR} ..."
    # Blobless: Arena's history is gigabytes; the checkout fetches only this commit's files. Its LFS
    # files are docs images and test data, so their pointers stay as they are (no git-lfs needed).
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
      https://github.com/isaac-sim/IsaacLab-Arena.git "${ARENA_DIR}" || return 1
    GIT_LFS_SKIP_SMUDGE=1 git -C "${ARENA_DIR}" checkout -q "${ARENA_REV}" || return 1
  elif [[ "${MANAGED}" == true && "$(git -C "${ARENA_DIR}" rev-parse HEAD)" != "${ARENA_REV}" ]]; then
    # ARENA_REV was bumped: move the managed checkout and re-sync (Arena's lock and Isaac Lab pin change with it).
    echo "setup.sh: moving IsaacLab-Arena to ${ARENA_REV:0:9} ..."
    GIT_LFS_SKIP_SMUDGE=1 git -C "${ARENA_DIR}" fetch -q origin "${ARENA_REV}" || return 1
    GIT_LFS_SKIP_SMUDGE=1 git -C "${ARENA_DIR}" checkout -q "${ARENA_REV}" || return 1
    GIT_LFS_SKIP_SMUDGE=1 git -C "${ARENA_DIR}" submodule update --depth 1 -- submodules/IsaacLab || return 1
    FORCE=true
  fi
  # Isaac Lab is built from Arena's submodule. Arena's .gitmodules points at SSH URLs; use HTTPS.
  if [[ ! -f "${ARENA_DIR}/submodules/IsaacLab/pyproject.toml" ]]; then
    echo "setup.sh: fetching Arena's Isaac Lab submodule ..."
    git -C "${ARENA_DIR}" submodule init -- submodules/IsaacLab || return 1
    git -C "${ARENA_DIR}" config submodule.submodules/IsaacLab.url https://github.com/isaac-sim/IsaacLab.git || return 1
    GIT_LFS_SKIP_SMUDGE=1 git -C "${ARENA_DIR}" submodule update --depth 1 -- submodules/IsaacLab || return 1
  fi

  if [[ "${FORCE}" == true || ! -x "${VENV}/bin/python" ]]; then
    echo "setup.sh: uv sync in ${ARENA_DIR} (the first run downloads Isaac Sim, about 10 GB) ..."
    (cd "${ARENA_DIR}" && uv sync) || return 1
    SYNCED=true
  fi

  # uv sync removes what Arena's lock doesn't list, so arena-so101 goes back in after every sync.
  if [[ "${SYNCED}" == true || -n "${EXTRAS}" ]] || ! "${VENV}/bin/python" -c 'import arena_so101' 2>/dev/null; then
    echo "setup.sh: installing arena-so101${EXTRAS} editable from ${HERE}/../.. ..."
    uv pip install --python "${VENV}/bin/python" -e "${HERE}/../..${EXTRAS}" || return 1
  fi

  # shellcheck disable=SC1091
  source "${VENV}/bin/activate" || return 1
  # Isaac Lab's scripts import `so101_table` by name (--external_environment_class_path); a script's own
  # directory is on sys.path, the working directory is not.
  case ":${PYTHONPATH:-}:" in
    *":${HERE}:"*) ;;
    *) export PYTHONPATH="${HERE}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
  echo "setup.sh: ready — python=$(command -v python), Arena ${ARENA_REV:0:9} in ${ARENA_DIR}"
}

# Refuse bare execution: activation must apply to the caller's shell.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  echo "setup.sh: source this script instead of executing it:" >&2
  echo "  source ./setup.sh" >&2
  exit 1
fi

_setup_main "$@"
_setup_rc=$?
unset -f _setup_main
return "${_setup_rc}"
