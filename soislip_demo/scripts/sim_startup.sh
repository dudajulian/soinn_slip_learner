#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$DEMO_PKG_DIR/../../.." && pwd)"
SESSION="soislip_husky"

# Load workspace environment once so package-share paths can be resolved.
set +u
source ~/.bashrc
source "$WORKSPACE_ROOT/install/setup.bash"
set -u

# LOG_DIR="$WORKSPACE_ROOT/logs"
# mkdir -p "$LOG_DIR"

DEMO_SHARE="$(ros2 pkg prefix soislip_demo)/share/soislip_demo"

ROBOT_RESOURCE_DIR="$DEMO_SHARE/resources/husky"
ELEVATION_CONFIG_DIR="$DEMO_SHARE/config/elevation_mapping"
RVIZ_CONFIG_FILE="$DEMO_SHARE/config/rviz/custom_rviz2.rviz"

declare -A cmds=(
  [coppelia_sim]="'$COPPELIASIM_ROOT_DIR'/coppeliaSim \
    '$ROBOT_RESOURCE_DIR'/scene.ttt -s 0 "
#     > '$LOG_DIR/coppelia_sim.log' 2>&1"
  [coppelia_controller]="ros2 launch coppelia_ros2_control coppelia_control.launch.py \
    controller_name:=platform_velocity_controller \
    controller_config_path:='$ROBOT_RESOURCE_DIR/control.yaml' \
    robot_description_path:='$ROBOT_RESOURCE_DIR/robot.urdf' "
#     > '$LOG_DIR/coppelia_controller.log' 2>&1"
  [elevation_mapping]="ros2 run elevation_mapping elevation_mapping \
    --ros-args \
    --params-file '$ELEVATION_CONFIG_DIR/sim_robot.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/elevation_map.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/aslam.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/postprocessor_pipeline.yaml' "
#     > '$LOG_DIR/elevation_mapping.log' 2>&1"
  [soislip_demo]="ros2 launch soislip_core soislip.launch.py \
    params_file:='$DEMO_SHARE/config/sim_params.yaml' "
#     > '$LOG_DIR/soislip_demo.log' 2>&1"
  [rviz]="ros2 run rviz2 rviz2 \
    --display-config '$RVIZ_CONFIG_FILE'"
  [teleop_key]="ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -p stamped:=true -p use_sim_time:=true \
    --remap cmd_vel:=/platform_velocity_controller/cmd_vel" 
  [teleop_joy]="ros2 launch teleop_twist_joy teleop-launch.py \
    joy_config:='xbox' publish_stamped_twist:=true use_sim_time:=true \
    joy_vel:=/platform_velocity_controller/cmd_vel" 
)

# Keep startup order aligned with all.launch.py.
startup_order=(
  coppelia_sim
  coppelia_controller
  elevation_mapping
  soislip_demo
  rviz
#   teleop_key
  teleop_joy
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

source_cmd="source ~/.bashrc; source '$WORKSPACE_ROOT/install/setup.bash'; cd '$WORKSPACE_ROOT'"

tmux new-session -d -s "$SESSION" -n "control"
for name in "${startup_order[@]}"; do
  echo "Starting $name ..."
  tmux new-window -t "$SESSION" -n "$name"
  tmux send-keys -t "$SESSION:$name" \
    "bash -lc 'set +u; $source_cmd; set -u; ${cmds[$name]}'" C-m
  sleep 2
done

echo "All windows ready. Attach with: tmux attach-session -t $SESSION"
tmux attach-session -t "$SESSION"