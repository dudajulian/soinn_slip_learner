# soislip_core

SOINN-based slip learning package for ROS 2, with C++ orchestration/feature nodes and Python model nodes.

## Installation
System Dependencies:
```bash
sudo apt update & sudo apt install python3-networkx 
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
echo "export CMAKE_PREFIX_PATH=/usr/local:#CMAKE_PREFIX_PATH" >> ~/.bashrc
source ~/.bashrc
```

Then build your workspace with:
```bash
colcon build \
    --merge-install \
    --parallel-workers 2 \
        --packages-up-to soislip_core\
    --cmake-args "-DCMAKE_BUILD_TYPE=Release" "-DCMAKE_EXPORT_COMPILE_COMMANDS=On"
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
        - Computes slip from TF motion (`odom` vs `map` reference)
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
2. `robot_experience_collector_node` computes slip from TF interval.
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
        - `wheel_separation`, `robot_frame`, `odom_frame`, `reference_frame`
        - `movement_threshold`, `collector_period_sec`
        - `sample_topic`, `feature_service_name`

- `soinn_training_node`
        - `model_path`, `init_new_model`, `sample_topic`
        - `input_dimension`, `max_edge_age`, `auto_save_period_sec`

- `soinn_prediction_node`
        - `model_path`, `service_name`, `feature_dim`, `model_reload_period_sec`

- `slip_prediction_manager`
        - `slip_prediction_map_topic`, `elevation_map_topic`
        - `slip_layer_name`, `confidence_layer_name`
        - `map_feature_service_name`, `predict_batch_service_name`, `prediction_period_sec`
