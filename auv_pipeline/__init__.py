"""AUV vision pipeline.

Stages (each is a subcommand of ``python -m auv_pipeline``):

    check      dataset sanity check (data.yaml, labels, class balance)
    augment    preview the underwater augmentation profiles
    train      config-driven YOLO training with underwater augmentation
    prune      structured channel pruning + recovery fine-tune
    export     ONNX / TensorRT (FP16 or INT8) export for the Jetson
    benchmark  mAP / params / latency comparison across models
    run        all of the above, end to end
"""

__version__ = "2.0.0"
