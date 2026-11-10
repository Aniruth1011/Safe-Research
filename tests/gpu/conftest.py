def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: requires a CUDA GPU, OpenVLA weights and the vla-safe/openvla fork")
