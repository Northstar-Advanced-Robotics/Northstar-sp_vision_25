#!/bin/bash
# Enter (creating if needed) the sp_vision_25 dev container on a Jetson Orin,
# with GPU access and a display connected to the Jetson's own X server.
#
# Usage: ./docker_enter_jetson.sh [--headless]
# Run from the repo root (or anywhere -- it resolves paths relative to this
# script's own location, not the current directory).
#
# --headless: for a pure-SSH session with no X server reachable at all (no
# monitor attached, no Xvfb, not even a logged-in desktop session) -- skips
# all of the DISPLAY/X11-auth handling below entirely and, when creating a
# FRESH container, skips wiring up X11 (DISPLAY/XAUTHORITY env, the
# /tmp/.X11-unix mount) at all. Fine for building, running the test
# binaries' own `--headless` modes (see JETSON_ORIN.md §6.2/6.4 --
# mt_detector_video_test's `--headless` flag needs zero X server, works
# with DISPLAY/XAUTHORITY completely unset), and any other CLI-only work.
# You will NOT be able to see `cv::imshow` windows from a container created
# this way without recreating it via a plain (non-headless) run of this
# script from a session that does have a working display.
#
# Without --headless, this script is meant to "just work" with zero extra
# commands: run it, land in a shell, `cv::imshow` windows show up on the
# Jetson's own monitor. That requires two things sorted out automatically
# every single time (not just at container-creation time), covered below.
#
# 1) DISPLAY auto-detection: if $DISPLAY isn't set in this shell (e.g. an
# SSH session with no X forwarding), the first display found under
# /tmp/.X11-unix is used instead -- this is the Jetson's own logged-in
# desktop session in the common single-seat case. Set DISPLAY explicitly
# yourself first if you have multiple X servers and want a specific one.
#
# 2) X11 auth: GDM (this repo's target desktop) manages its own
# per-session auth cookie at /run/user/<uid>/gdm/Xauthority, separate from
# ~/.Xauthority and regenerated fresh on every login/reboot -- confirmed by
# checking the actual running Xorg process's `-auth` argument. A cookie
# bind-mounted once at container-*creation* time goes stale the moment
# that regenerates, and `docker start`/`docker restart` doesn't refresh a
# bind-mounted *file* even though the container itself survives fine (see
# JETSON_ORIN.md §6.5 for the underlying investigation). So instead of a
# bind mount, this script `docker cp`s whichever auth file is actually live
# right now (preferring the GDM per-session one, falling back to
# ~/.Xauthority if that's not present -- e.g. a non-GDM desktop) into the
# container on EVERY run, then passes DISPLAY/XAUTHORITY explicitly on the
# `docker exec` itself -- so a stale value baked in at creation time never
# matters, whether this is a fresh container, a restarted one, or one
# that's just been sitting there running since your last login.
#
# The container is created once and reused (not --rm) so that the compiled
# build/ directory and cached TensorRT engine (assets/yolov5.engine) persist
# across sessions instead of rebuilding from scratch every time.
#
# The ENTIRE repo is bind-mounted (not just configs/logs/assets) so the
# container always sees whatever's actually on the host's disk right now --
# git pull, branch switches, local edits, all show up immediately with no
# rebuild or manual file-copying needed. Same pattern the Windows/x86_64
# devcontainer.json already uses. Without this, only the source tree that
# existed at whatever moment `docker build` last ran stays baked into the
# image, silently diverging from the host repo -- easy to not notice.
#
# --privileged -v /dev:/dev gives the container access to the host's USB
# camera, CAN, and serial device nodes (/dev/videoN, /dev/canN, /dev/ttyUSBN,
# ...) -- see JETSON_ORIN.md §5.2. Without it, /dev/video0 etc. simply don't
# exist inside the container, so e.g. camera_test's V4L2 backend fails to
# open the device at all (OpenCV's error in that case is the unhelpful
# "backend is generally available but can't be used to capture by name").
#
# QT_X11_NO_MITSHM=1 disables the X MIT-SHM (shared memory) extension for
# Qt's own windows, but NOT GTK's (OpenCV highgui here uses the GTK backend,
# per the "Failed to load module canberra-gtk-module" log line) -- GTK can
# still crash with "BadShmSeg (invalid shared segment parameter)" without
# --ipc=host. Root cause: containers get their own IPC namespace by default,
# so SysV/POSIX shared-memory segments a process creates *inside* the
# container for MIT-SHM XImages aren't visible to the X server running on
# the host, outside that namespace -- the segment ID the client references
# simply doesn't exist from the server's point of view. --ipc=host shares
# the host's IPC namespace instead, fixing this at the root (matches how
# NVIDIA's own reference ROS container commands set up X11 too).

set -e
 
CONTAINER_NAME="sp_vision_jetson"
IMAGE_NAME="sp_vision_25"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HEADLESS=0
for arg in "$@"; do
  case "$arg" in
    --headless) HEADLESS=1 ;;
    *)
      echo "Unknown argument: $arg (only --headless is recognized)" >&2
      exit 1
      ;;
  esac
done

if [ "$HEADLESS" -eq 0 ]; then
  if [ -z "$DISPLAY" ]; then
    detected_socket="$(ls /tmp/.X11-unix/ 2>/dev/null | head -1)"
    if [ -n "$detected_socket" ]; then
      DISPLAY=":${detected_socket#X}"
      echo "DISPLAY not set in this shell -- auto-detected ${DISPLAY} from /tmp/.X11-unix." >&2
    else
      echo "DISPLAY is not set and no X server socket was found under /tmp/.X11-unix." >&2
      echo "If you don't need any GUI window at all, use --headless instead." >&2
      exit 1
    fi
  fi

  XAUTH_SRC="/run/user/$(id -u)/gdm/Xauthority"
  if [ ! -f "$XAUTH_SRC" ]; then
    XAUTH_SRC="$HOME/.Xauthority"
  fi
  if [ ! -f "$XAUTH_SRC" ]; then
    echo "Warning: no X11 auth cookie found (checked /run/user/$(id -u)/gdm/Xauthority" >&2
    echo "and $HOME/.Xauthority) -- X11 auth may fail." >&2
    echo "Falling back to 'xhost +local:docker' may be needed instead; see JETSON_ORIN.md." >&2
  fi
fi

if sudo docker ps -q -f name="^${CONTAINER_NAME}\$" | grep -q .; then
  echo "Container '${CONTAINER_NAME}' already running, attaching..."
elif sudo docker ps -aq -f name="^${CONTAINER_NAME}\$" | grep -q .; then
  echo "Starting existing (stopped) container '${CONTAINER_NAME}'..."
  sudo docker start "${CONTAINER_NAME}" >/dev/null
elif [ "$HEADLESS" -eq 1 ]; then
  echo "Creating container '${CONTAINER_NAME}' (--headless: no X11 wiring)..."
  sudo docker run -d --name "${CONTAINER_NAME}" \
    --runtime nvidia \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "${REPO_DIR}:/root/sp_vision_25" \
    -v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro \
    --privileged \
    -v /dev:/dev \
    "${IMAGE_NAME}" \
    sleep infinity
else
  echo "Creating container '${CONTAINER_NAME}'..."
  sudo docker run -d --name "${CONTAINER_NAME}" \
    --runtime nvidia \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e QT_X11_NO_MITSHM=1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${REPO_DIR}:/root/sp_vision_25" \
    -v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro \
    --privileged \
    -v /dev:/dev \
    "${IMAGE_NAME}" \
    sleep infinity
fi

if [ "$HEADLESS" -eq 0 ] && [ -f "$XAUTH_SRC" ]; then
  sudo docker cp "$XAUTH_SRC" "${CONTAINER_NAME}:/root/.Xauthority"
fi

echo "Entering container. First time in a fresh container, build with:"
echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
if [ "$HEADLESS" -eq 1 ]; then
  sudo docker exec -it "${CONTAINER_NAME}" bash
else
  sudo docker exec -it -e DISPLAY="${DISPLAY}" -e XAUTHORITY=/root/.Xauthority "${CONTAINER_NAME}" bash
fi
