#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$DEMO_PKG_DIR/../../.." && pwd)"
SESSION="soislip_husky"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$WORKSPACE_ROOT/resources/results"
SAMPLE_CSV="$RESULTS_DIR/${RUN_TS}_sim_samples.csv"

# Load workspace environment once so package-share paths can be resolved.
set +u
source ~/.bashrc
source "$WORKSPACE_ROOT/install/setup.bash"
set -u

# LOG_DIR="$WORKSPACE_ROOT/logs"
# mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

DEMO_SHARE="$(ros2 pkg prefix soislip_demo)/share/soislip_demo"

ROBOT_RESOURCE_DIR="$DEMO_SHARE/resources/husky"
RVIZ_CONFIG_FILE="$DEMO_SHARE/config/rviz/sim_husky.rviz"
ELEVATION_CONFIG_DIR="$DEMO_PKG_DIR/config/elevation_mapping"
SOISLIP_CONFIG_FILE="$DEMO_PKG_DIR/config/sim_params.yaml"

declare -A cmds=(
  [coppelia_sim]="$COPPELIASIM_ROOT_DIR/coppeliaSim \
    $ROBOT_RESOURCE_DIR/scene.ttt -s 0 "
#     > '$LOG_DIR/coppelia_sim.log' 2>&1"
  [coppelia_controller]="ros2 launch coppelia_ros2_control coppelia_control.launch.py \
    controller_name:=platform_velocity_controller \
    controller_config_path:='$ROBOT_RESOURCE_DIR/control.yaml' \
    robot_description_path:='$ROBOT_RESOURCE_DIR/robot.urdf' \
    use_sim_time:=true "
#     > '$LOG_DIR/coppelia_controller.log' 2>&1"
  [elevation_mapping]="ros2 run elevation_mapping elevation_mapping \
    --ros-args \
    --params-file '$ELEVATION_CONFIG_DIR/sim_robot.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/elevation_map.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/aslam.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/postprocessor_pipeline.yaml' "
#     > '$LOG_DIR/elevation_mapping.log' 2>&1"
  [soislip_demo]="ros2 launch soislip_core soislip.launch.py \
    params_file:='$SOISLIP_CONFIG_FILE'" 
#     > '$LOG_DIR/soislip_demo.log' 2>&1"
  [rviz]="ros2 run rviz2 rviz2 \
    --display-config '$RVIZ_CONFIG_FILE'
    --ros-args -p use_sim_time:=true "
  [teleop_key]="ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -p stamped:=true -p use_sim_time:=true -p speed:=0.5 -p turn:=0.3\
    --remap cmd_vel:=/platform_velocity_controller/cmd_vel" 
  [teleop_joy]="ros2 launch teleop_twist_joy teleop-launch.py \
    joy_config:='xbox' publish_stamped_twist:=true use_sim_time:=true \
    joy_vel:=/platform_velocity_controller/cmd_vel" 
  [auto_cmdvel]="ros2 topic pub /platform_velocity_controller/cmd_vel \
    geometry_msgs/msg/TwistStamped '{header: {stamp: {sec: 0, nanosec: 0}, \
    frame_id: \"base_link\"}, twist: {linear: {x: 0.5, y: 0.0, z: 0.0}, \
    angular: {x: 0.0, y: 0.0, z: 0.15}}}' -r 10"
  [sample_recorder]="ros2 run soislip_core sample_recorder_node.py \
    --ros-args -p use_sim_time:=true \
    -p sample_topic:=/experience_samples \
    -p output_csv_path:='$SAMPLE_CSV'"
)

# Keep startup order aligned with all.launch.py.
startup_order=(
  coppelia_sim
  coppelia_controller
  elevation_mapping
  soislip_demo
  rviz
  sample_recorder
  teleop_key
  # teleop_joy
  # auto_cmdvel
)

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

source_cmd="source ~/.bashrc; source $WORKSPACE_ROOT/install/setup.bash; cd $WORKSPACE_ROOT"


tmux new-session -d -s "$SESSION" -n "control"
tmux set-option -t "$SESSION" mouse on
for name in "${startup_order[@]}"; do
  echo "Starting $name ..."
  tmux new-window -t "$SESSION" -n "$name"
  full_cmd="set +u; $source_cmd; set -u; ${cmds[$name]}"
  tmux send-keys -t "$SESSION:$name" \
    "bash -lc $(printf '%q' "$full_cmd")" C-m
  sleep 2
done

echo "All windows ready. Attach with: tmux attach-session -t $SESSION"
tmux attach-session -t "$SESSION"