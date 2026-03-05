# soinn_slip_learner

SOINN-based slip learning package for ROS 2, with C++ orchestration/feature nodes and Python model nodes.

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
        - Calls `get_map_features`
        - Calls `predict_batch`
        - Publishes flat `(x, y, pred)` triples on `/gridmap_with_predictions`
        - Output format: `[x0, y0, p0, x1, y1, p1, ...]`

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
        - Response: `predictions`, `success`, `message`

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
3. Manager publishes mapped predictions on `/gridmap_with_predictions`.

## Launch files

In `launch/`:

- `training.launch.py` – training-oriented stack
- `prediction.launch.py` – prediction-oriented stack
- `full_system.launch.py` – all package nodes
- `all.launch.py` – includes:
        - `coppelia_ros2_control`
        - `elevation_mapping`
        - `full_system.launch.py`

Run:

```bash
ros2 launch soinn_slip_learner all.launch.py
```

## Parameters

Main parameters are in `config/soinn_params.yaml`:

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
        - `output_topic`, `map_feature_service_name`, `predict_batch_service_name`, `prediction_period_sec`

## Repository structure (high level)

```bash
soinn_slip_learner/
├── CMakeLists.txt
├── package.xml
├── config/
├── launch/
├── models/
├── soinn_slip_learner/     # Python nodes and SOINN implementation
├── src/                    # C++ nodes
└── srv/                    # ROS service definitions
```
