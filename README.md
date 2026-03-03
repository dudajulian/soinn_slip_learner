# soinn_slip_learner

### **File Structure**
```bash
soinn_slip_learner/
├── CMakeLists.txt                # For C++ nodes
├── package.xml
├── launch/
│   ├── training.launch.py
│   ├── prediction.launch.py
│   └── full_system.launch.py
├── config/
│   ├── soinn_params.yaml
│   └── feature_extractor.yaml
├── include/soinn_slip_learner/   # C++ headers
├── src/                          # C++ source files
│   ├── gridmap_feature_extractor.cpp
│   ├── robot_experience_collector.cpp
│   └── slip_prediction_manager.cpp
├── soinn_slip_learner/           # Python scripts
│   ├── latent_feature_extractor_node.py
│   ├── feature_aggregator_node.py
│   ├── soinn_training_node.py
│   └── soinn_prediction_node.py
├── models/
│   └── latent_feature_model.onnx
└── README.md
```

---

### **Node Architecture**
#### **C++ Nodes** (Performance-critical)
| Node Name                          | Purpose                                                                                     | Inputs                                  | Outputs                                 |
|------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------|-----------------------------------------|
| `gridmap_feature_extractor_node`   | Extracts handcrafted features (elevation, roughness, etc.) from gridmap.                     | `/gridmap`                              | Services: `get_cell_features`, `get_grid_features` |
| `robot_experience_collector_node`  | Collects (feature, slip) pairs for SOINN training (uses your existing C++ slip calculation).   | `/tf`, `/odometry`, `get_cell_features_combined` | `/experience_samples`                   |
| `slip_prediction_manager`         | Orchestrates feature extraction and prediction for the full gridmap.                      | Service: `get_grid_features_combined`, `predict_batch` | `/gridmap_with_predictions`             |

#### **Python Nodes** (ML/SOINN or feature aggregation)
| Node Name                          | Purpose                                                                                     | Inputs                                  | Outputs                                 |
|------------------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------|-----------------------------------------|
| `latent_feature_extractor_node`    | Extracts latent features using a pretrained model.                                         | `/gridmap`                              | Services: `get_cell_latent_features`, `get_grid_latent_features` |
| `feature_aggregator_node`          | Combines handcrafted (C++) and latent (Python) features for cells/gridmaps.                 | Services: `get_cell_features`, `get_cell_latent_features`, `get_grid_features`, `get_grid_latent_features` | Services: `get_cell_features_combined`, `get_grid_features_combined` |
| `soinn_training_node`              | Trains the SOINN model with experience samples (combined features + slip).                   | `/experience_samples`                  | (Updated SOINN model)                   |
| `soinn_prediction_node`            | Provides batch prediction of slip using SOINN.                                             | Service: `predict_batch`               | Predicted slip values                   |

---

### **Data Flow**
#### **1. Training Workflow**
```
/gridmap → gridmap_feature_extractor_node (C++) --(handcrafted)-->
                latent_feature_extractor_node (Python) --(latent)-->
                        feature_aggregator_node (Python) --(combined)-->
                                robot_experience_collector_node (C++) --(experience_samples)-->
                                        soinn_training_node (Python)
```

#### **2. Prediction Workflow**
```
/gridmap → gridmap_feature_extractor_node (C++) --(handcrafted)-->
                latent_feature_extractor_node (Python) --(latent)-->
                        feature_aggregator_node (Python) --(combined)-->
                                slip_prediction_manager (C++) --(predictions)-->
                                        /gridmap_with_predictions
```

---

### **Services**
| Service Name                     | Input                          | Output                          | Provider Node                     |
|-----------------------------------|--------------------------------|---------------------------------|-----------------------------------|
| `get_cell_features`               | `geometry_msgs/Point` cell index | `float32[]` handcrafted features | `gridmap_feature_extractor_node`  |
| `get_grid_features`               | `std_msgs/Header`              | `float32[]` handcrafted features | `gridmap_feature_extractor_node`  |
| `get_cell_latent_features`        | `geometry_msgs/Point` cell index | `float32[]` latent features     | `latent_feature_extractor_node`   |
| `get_grid_latent_features`        | `std_msgs/Header`              | `float32[]` latent features     | `latent_feature_extractor_node`   |
| `get_cell_features_combined`      | `geometry_msgs/Point` cell index | `float32[]` combined features   | `feature_aggregator_node`         |
| `get_grid_features_combined`      | `std_msgs/Header`              | `float32[]` combined features   | `feature_aggregator_node`         |
| `predict_batch`                   | `float32[]` features           | `float32[]` predictions         | `soinn_prediction_node`           |
