# Frozen feature extraction

The adapted OpenVLA repository is external to SAFE. Extraction must run the frozen backbone in inference mode, capture configured hidden-state layers in one forward pass where possible, apply a documented pooling operation, and preserve observation/action/label alignment. Store the result with `FeatureStore` metadata. The detector CLI consumes cached features and never loads OpenVLA when the cache exists.

Layer IDs are validated against the declared `num_layers` in metadata. The framework does not assume the final layer index; use the extraction metadata from the loaded model. The current synthetic fixtures use explicit zero-based IDs only as test data, not as a claim about OpenVLA depth.
