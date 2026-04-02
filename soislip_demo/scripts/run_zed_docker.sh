#!/usr/bin/env bash

# Determine container name (does not support suffixes)
PLATFORM="$(uname -m)"
CONTAINER_NAME="isaac_ros_dev-${PLATFORM}-container"


# Start the exited container and run the script.
if [ "$(docker ps -a --quiet --filter status=exited --filter name=${CONTAINER_NAME})" ]; then
    docker start ${CONTAINER_NAME}
    # ISAAC_ROS_WS=$(docker exec ${CONTAINER_NAME} printenv ISAAC_ROS_WS)
    # echo "Docker workspace: ${ISAAC_ROS_WS}"

    # docker exec -i -t -u admin --workdir ${ISAAC_ROS_WS} ${CONTAINER_NAME} /bin/bash -lc "source ~/.bashrc && ./start_zed_node.sh"
    # exit 0
fi

# Attach to the container, or start it if it doesn't exist.
cd ${ISAAC_ROS_WS}/src/isaac_ros_common && \
./scripts/run_dev.sh -i ros2_humble.zed \
-a "-v /usr/local/zed/settings:/usr/local/zed/settings \
    -v /usr/local/zed/resources:/usr/local/zed/resources" 
# --  -lc './source_zed.sh && ./start_zed_node.sh; exec bash'
