#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_PKG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_ROOT="$(cd "$DEMO_PKG_DIR/../../.." && pwd)"
SESSION="soislip_husky"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$WORKSPACE_ROOT/results"
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

ROBOT_RESOURCE_DIR="$DEMO_PKG_DIR/resources/husky"
ELEVATION_CONFIG_DIR="$DEMO_PKG_DIR/config/elevation_mapping"
RVIZ_CONFIG_FILE="$DEMO_PKG_DIR/config/rviz/husky.rviz"

declare -A cmds=(
  [zed_node]="ros2 launch zed_wrapper zed_camera_tf_remap.launch.py \
    camera_model:=zed2 \
    publish_tf:=false \
    tf_topic:=/tug_husky/tf \
    tf_static_topic:=/tug_husky/tf_static \
    namespace:=jetson"
  [static_tf_zed]="ros2 run tf2_ros static_transform_publisher \
    0.41 0.0 0.39 0 0.0 0 base_link zed_camera_link \
    --ros-args -r /tf:=/tug_husky/tf -r /tf_static:=/tug_husky/tf_static"
  [laser_odom]="ros2 launch rf2o_laser_odometry rf2o_laser_odometry.launch.py laser_scan_topic:=/scan publish_tf:=False"
  [camera_odom_republisher]="ros2 run soislip_core camera_odom_republisher --ros-args \
    -p input_topic:=/odom_rf2o -p output_topic:=/visual_odom -p base_frame:=base_link \
    -p publish_tf:=true -r /tf:=/tug_husky/tf -r /tf_static:=/tug_husky/tf_static"
  [elevation_mapping]="ros2 run elevation_mapping elevation_mapping \
    --ros-args -r /tf:=/tug_husky/tf -r /tf_static:=/tug_husky/tf_static \
    --params-file '$ELEVATION_CONFIG_DIR/zed2_robot.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/elevation_map.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/aslam.yaml' \
    --params-file '$ELEVATION_CONFIG_DIR/postprocessor_pipeline.yaml'"
#     > '$LOG_DIR/elevation_mapping.log' 2>&1"
  [soislip_demo]="ros2 launch soislip_core soislip.launch.py \
    tf_topic:=/tug_husky/tf \
    tf_static_topic:=/tug_husky/tf_static \
    params_file:='$DEMO_PKG_DIR/config/params.yaml'"
#     > '$LOG_DIR/soislip_demo.log' 2>&1"
  [rviz]="ros2 run rviz2 rviz2 \
    --display-config '$RVIZ_CONFIG_FILE' \
    --ros-args -r /tf:=/tug_husky/tf -r /tf_static:=/tug_husky/tf_static"
  [teleop_key]="ros2 run teleop_twist_keyboard teleop_twist_keyboard \
    --ros-args -p stamped:=false \
    --remap cmd_vel:=/tug_husky/cmd_vel" 
  [teleop_joy]="ros2 launch teleop_twist_joy teleop-launch.py \
    joy_config:='xbox' publish_stamped_twist:=false  \
    joy_vel:=/tug_husky/cmd_vel" 
  [sample_recorder]="ros2 run soislip_core sample_recorder_node.py \
    -p sample_topic:=/experience_samples \
    -p output_csv_path:='$SAMPLE_CSV'"
)

# Keep startup order aligned with all.launch.py.
startup_order=(
  # static_tf_zed
  # zed_node
  laser_odom
  camera_odom_republisher
  # elevation_mapping
  # soislip_demo
  # rviz
  teleop_key
  # teleop_joy
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
