# soislip_demo
> NOTE: for all general instructions check the `README.md` of the soinn_slip_learner repository.

## Customization

### `resources/husky/zenoh_config.json5`
This config works for all devices
- mode: "peer" (Assuming Jetson, Husky and BaseStation are in the same Subnet)
- transient_local_cache_multiplier: 100 (To avoid loosing static transform)