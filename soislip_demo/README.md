# soislip_demo
> NOTE: for all general instructions check the `README.md` of the soinn_slip_learner repository.

## Customization

### `resources/zed_camera_tf_remap.launch`
By default the zed_node publishes to /tf and /tf_static no matter the namespace so I added two parameters to the launch file so one can remap these (usefull for all clearpath robots)