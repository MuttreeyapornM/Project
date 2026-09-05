#!/usr/bin/env bash
# Run navigate.py with the right interpreter and environment.
#
#   ./run_navigate.sh --use_camera --front_cam 2 --left_cam 4 --rear_cam 0 --right_cam 7 \
#                     --ckpt checkpoints/iter_27000_deeplabv3plus_mobilenet_custom_os16.pth
#   ./run_navigate.sh --input video5fps1.mp4 --output test_results \
#                     --ckpt checkpoints/iter_27000_deeplabv3plus_mobilenet_custom_os16.pth
#
# Three things this gets right that a bare `python3 navigate.py` does not:
#   1. venv/bin/python  -- the system python3 has no torch and no kornia
#   2. ROS 2 Foxy sourced -- navigate.py imports geometry_msgs
#   3. DISPLAY set -- navigate.py calls cv2.imshow unconditionally
set -eo pipefail
cd "$(dirname "$0")"

if [ ! -x ./venv/bin/python ]; then
  echo "error: ./venv/bin/python not found. Run from ~/model/main." >&2; exit 1
fi
# shellcheck disable=SC1091
set +u
source /opt/ros/foxy/setup.bash
set -u
export DISPLAY="${DISPLAY:-:1}"

if ! xset q >/dev/null 2>&1; then
  echo "warning: no usable X display on '$DISPLAY'." >&2
  echo "         navigate.py calls cv2.imshow and will crash without one." >&2
  echo "         Run it from the Xavier's own desktop, or ssh -X, or set DISPLAY=:1." >&2
fi

exec ./venv/bin/python run_navigate.py "$@"
