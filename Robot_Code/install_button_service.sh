#!/usr/bin/env bash
# Make button_start.py run by itself every time the Pi boots, so the robot needs no
# keyboard, screen or SSH: power on, wait for the yellow blip, press the button.
#
#   ./install_button_service.sh                  # install + start now
#   ./install_button_service.sh --test-mode      # same, and main.py always gets --test-mode
#   ./install_button_service.sh --remove         # uninstall
#
# Run it WITHOUT sudo, from the same shell you normally run main.py from (with your
# venv activated, if you use one) -- it records which python3 that is and which user
# you are, and asks for sudo itself only to write the system file.
#
# Afterwards:
#   tail -f runs/button_service.log              # watch what the robot is doing
#   sudo systemctl stop robot-button             # stop it (e.g. to run main.py by hand)
#   sudo systemctl start robot-button            # start it again
#   sudo systemctl status robot-button           # is it running?

set -euo pipefail

NAME=robot-button
UNIT=/etc/systemd/system/$NAME.service
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--remove" ]]; then
    sudo systemctl disable --now "$NAME" 2>/dev/null || true
    sudo rm -f "$UNIT"
    sudo systemctl daemon-reload
    echo "removed $NAME -- main.py no longer starts from the button at boot"
    exit 0
fi

if [[ $EUID -eq 0 ]]; then
    echo "run this WITHOUT sudo (it would record root's python3 and user, not yours)" >&2
    exit 1
fi

PY="$(command -v python3)"
USER_NAME="$(id -un)"
EXTRA=""
if [[ $# -gt 0 ]]; then
    EXTRA=" -- $*"
fi

# The service runs as you, so you need the hardware groups (on by default for the
# first user on Pi OS, but a second account may not have them).
for g in gpio video i2c; do
    if getent group "$g" >/dev/null && ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$g"; then
        echo "adding $USER_NAME to the $g group (needed for the pins / camera / motor board)"
        sudo usermod -aG "$g" "$USER_NAME"
    fi
done

mkdir -p "$HERE/runs"
chmod +x "$HERE/button_start.py"

sudo tee "$UNIT" >/dev/null <<EOF
[Unit]
Description=EGB320 robot: run main.py when the button is pressed
After=multi-user.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$HERE
Environment=PYTHONUNBUFFERED=1
ExecStart=$PY $HERE/button_start.py$EXTRA
# If the launcher itself dies, bring it back -- the button should always work.
Restart=on-failure
RestartSec=3
# 'stop' signals only the launcher (mixed), which then stops main.py cleanly itself.
KillMode=mixed
TimeoutStopSec=20
StandardOutput=append:$HERE/runs/button_service.log
StandardError=inherit

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$NAME"
sudo systemctl restart "$NAME"

echo
echo "installed: $NAME"
echo "  python : $PY"
echo "  user   : $USER_NAME"
echo "  runs   : $HERE/button_start.py$EXTRA"
echo
echo "The yellow LED should now blip every 2 s. Press the button to start main.py."
echo "Log: tail -f $HERE/runs/button_service.log"
