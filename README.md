<p align="left">
  <img src="./icon.png" alt="Repository icon" width="128" height="128">
</p>

# SOISLIP

SOINN-based slip learning package for ROS 2, with C++ orchestration/feature nodes and Python model nodes.

## Installation
System Dependencies:
```bash
sudo apt update && sudo apt install ros-humble-nav2-costmap-2d ros-humble-grid-map
```

From your workspace directory import dependency repositories:
```bash
vcs import src < src/soinn_slip_learner/.repos
```

The kindr package needs to be build with make instead of colon. To avoid building it with
later on, add a `COLCON_IGNORE` file to it.
```bash
cd src/kindr
touch COLCON_IGNORE
mkdir -p build
cd build
cmake .. -DUSE_CMAKE=true
sudo make install
export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH
cd ../../../
```
It is a good idea to add the CMAKE_PREFIX_PATH to your ~/.bashrc
```bash
echo "export CMAKE_PREFIX_PATH=/usr/local:$CMAKE_PREFIX_PATH" >> ~/.bashrc
source ~/.bashrc
```

Install the ROS Dependencies (Note that `kindr` and `coppelia_ros2_control` will not be found, this is expected and can be ignored)
```bash
rosdep install --from-paths src --ignore-src -y
```

Then build and source the packages from your workspace directory with:
```bash
colcon build \
    --parallel-workers $(nproc) \
        --packages-up-to soislip_demo\
    --cmake-args "-DCMAKE_BUILD_TYPE=Release"
source install/setup.bash
```

Launch the core package (see the example parameter file `soislip_demo/config/soislip_params.yaml`):
```bash
ros2 launch soislip_core soislip.launch.py params_file:=path/to/parameter_file
```

## Demo with Coppelia Sim and Husky Robot
To see the package in action launch the demo tmux session `soislip_demo/scripts/startup.sh`. For this to work you need to install and setup CoppeliaSim. Follow the instructions of [`coppelia_ros2_control`](https://github.com/dudajulian/coppelia_ros2_control) package which was developed along side with this project.
> NOTE: Per default the `startup.sh` launches a `teleop_twist_joy` node for controlling the robot with an XBOX 360 controller. If you prefer to use `teleop_twist_keyboard` instead simply uncomment it in the `startup_order` inside `startup.sh`.

Also make to set `COPPELIASIM_ROOT_DIR` correctly. (This should be the case if the `sim_ros2_interface` was installed correctly.)
```bash
export COPPELIASIM_ROOT_DIR=~/path/to/coppeliaSim/folder
```

## Demo with Real Robot
**Prerequisites:**
- Robot publishing wheel odometry
- Vision sensors on robot publishing pointcloud
- Reference odometry (e.g. visual odometry)
- Follow the network recommendations of [Stereolabs ZED](https://www.stereolabs.com/docs/ros2/dds-and-network-tuning). Also see the the `RMW_CYCLONEDDS_CONFIG.xml` here.

change into your workspace folder
```bash
export ROS_DOMAIN_ID=30
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=$(pwd)/src/soinn_slip_learner/soislip_demo/resources/husky/RMW_CYCLONEDDS_CONFIG.xml
export ZENOH_SESSION_CONFIG_URI=$(pwd)/src/soinn_slip_learner/soislip_demo/resources/husky/RMW_ZENOH_SESSION_CONFIG.json5
source $(pwd)/install/local_setup.bash
```

## Current architecture (implemented)

### C++ nodes

- `gridmap_feature_extractor_node`
        - Subscribes to `/elevation_map` (`grid_map_msgs/msg/GridMap`)
        - Caches latest map
        - Provides:
                - `get_cell_features` (`GetCellFeatures.srv`)
                - `get_map_features` (`GetMapFeatures.srv`)

- `robot_experience_collector_node`
        - Computes slip from either TF frames or `nav_msgs/msg/Odometry` topics
        - Queries `get_cell_features` at the midpoint position
        - Publishes training samples to `/experience_samples` (`std_msgs/msg/Float32MultiArray`)
        - Sample format: `[slip, f0, f1, ...]`

- `slip_prediction_manager`
        - Subscribes to `/elevation_map` (`grid_map_msgs/msg/GridMap`)
        - Calls `get_map_features`
        - Calls `predict_batch`
        - Publishes a `grid_map_msgs/msg/GridMap` on `/slip_prediction_map` with added `slip_prediction` and `slip_confidence` layers

### Python nodes

- `soinn_training_node.py`
        - Subscribes to `/experience_samples`
        - Trains `SoinnPlus`
        - Saves model periodically and on shutdown (atomic write)

- `soinn_prediction_node.py`
        - Provides `predict_batch`
        - Loads/reloads serialized model from disk
        - Predicts slip for feature batches

- `latent_feature_extractor_node.py`
        - Placeholder node (currently no feature extraction logic)

## Services

- `GetCellFeatures.srv`
        - Request: `geometry_msgs/Point position`
        - Response: `features`, `success`, `message`

- `GetMapFeatures.srv`
        - Request: empty
        - Response:
                - `features` (flattened feature vectors)
                - `positions` (`geometry_msgs/Point[]`, aligned with feature vectors)
                - `feature_dim`
                - `success`, `message`

- `PredictBatch.srv`
        - Request: `features` (flattened), `feature_dim`
        - Response: `predictions`, `confidence_scores`, `success`, `message`

## Data flow

### Training path

1. `gridmap_feature_extractor_node` keeps latest map.
2. `robot_experience_collector_node` computes slip from a frame- or topic-based odometry interval.
3. Collector queries `get_cell_features` at midpoint position.
4. Collector publishes `[slip + features]` to `/experience_samples`.
5. `soinn_training_node.py` trains and persists the model.

### Prediction path

1. `slip_prediction_manager` requests full map features via `get_map_features`.
2. Manager calls `predict_batch` on `soinn_prediction_node.py`.
3. Manager publishes a `GridMap` with prediction and confidence layers on `/slip_prediction_map`.

## Launch files

Launch files are located in the demo package.

Run Husky demo stack:

```bash
ros2 launch soislip_demo demo_husky.launch.py
```

## Parameters

Main parameters are in `config/soislip_params.yaml`:

- `gridmap_feature_extractor_node`
        - `elevation_map_topic`
        - `feature_service_name`
        - `map_feature_service_name`
        - `feature_radius`

- `robot_experience_collector_node`
        - `wheel_separation`, `robot_frame`, `wheel_odom`, `reference_odom`
        - `wheel_odom_source`, `reference_odom_source`, `odom_timeout_sec`
        - `movement_threshold`, `collector_period_sec`
        - `sample_topic`, `feature_service_name`

- `soinn_training_node`
        - `model_path`, `init_new_model`, `sample_topic`
        - `input_dimension`, `max_edge_age`, `auto_save_period_sec`

- `soinn_prediction_node`
        - `model_path`, `service_name`, `feature_dim`, `model_reload_period_sec`

> **Note — stale model on first prediction:** `soinn_prediction_node` loads whatever file exists at `model_path` as soon as it is available, without waiting for `soinn_training_node` to finish initialising a new model.
> If you change the input dimension or use-case and set `init_new_model: True` in `soinn_training_node`, but an old model file still exists at the same `model_path`, the prediction node will silently load that old file on startup and continue serving stale predictions.
> This can happen because `soinn_training_node` defers model creation until the first training sample arrives when `input_dimension: 0`, so the new model file may not exist yet at the time the prediction node first tries to load it.
> **To avoid this**, when resetting the model do one of the following before starting the nodes:
> - Change `model_path` in both `soinn_training_node` and `soinn_prediction_node` to a new path.
> - Delete or move the existing model file at the old `model_path`.

- `slip_prediction_manager`
        - `slip_prediction_map_topic`, `elevation_map_topic`
        - `slip_layer_name`, `confidence_layer_name`
        - `map_feature_service_name`, `predict_batch_service_name`, `prediction_period_sec`
