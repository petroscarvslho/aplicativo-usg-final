#!/usr/bin/env python3
"""
Registry central de plugins/modelos para treino e inferencia.
Mantem nomes esperados, tipos de tarefa e formatos de labels.
"""

from typing import Dict, Optional


PLUGIN_SPECS: Dict[str, Dict] = {
    "NEEDLE": {
        "default_model": "vasst",
        "models": {
            "vasst": {
                "task": "regression",
                "arch": "vasst",
                "expected_path": "models/vasst_needle.pt",
                "label_type": "point_yx",
                "label_scale": "0_1",
                "input_size": (256, 256),
                "channels": 1,
            },
            "yolo": {
                "task": "detection",
                "arch": "yolo",
                "expected_path": "models/best.pt",
                "label_type": "yolo_bbox",
                "input_size": (640, 640),
                "channels": 3,
                "class_names": ["needle"],
            },
        },
    },
    "NERVE": {
        "default_model": "unet",
        "models": {
            "unet": {
                "task": "segmentation",
                "arch": "smp_unet",
                "expected_path": "models/unet_nerve.pt",
                "label_type": "mask_classes",
                "num_classes": 3,  # background + nerve + vessel
                "input_size": (256, 256),
                "channels": 1,
            },
            "nerve_track": {
                "task": "segmentation",
                "arch": "nerve_track",
                "expected_path": "models/nerve_segmentation/nerve_unetpp_effb4.pt",
                "label_type": "mask_classes",
                "num_classes": 6,
                "input_size": (384, 384),
                "channels": 3,
            },
        },
    },
    "CARDIAC": {
        "default_model": "echonet",
        "models": {
            "echonet": {
                "task": "regression",
                "arch": "echonet",
                "expected_path": "models/echonet.pt",
                "label_type": "ef_esv_edv",
                "label_scale": "0_1",
                "input_size": (224, 224),
                "channels": 1,
            },
        },
    },
    "FAST": {
        "default_model": "yolo",
        "models": {
            "yolo": {
                "task": "detection",
                "arch": "yolo",
                "expected_path": "models/fast_detector.pt",
                "label_type": "yolo_bbox",
                "input_size": (640, 640),
                "channels": 3,
                "class_names": ["fluid"],
            },
        },
    },
    "ANATOMY": {
        "default_model": "usfm",
        "models": {
            "usfm": {
                "task": "segmentation",
                "arch": "smp_unet",
                "expected_path": "models/usfm_segment.pt",
                "label_type": "mask_classes",
                "num_classes": 5,
                "input_size": (256, 256),
                "channels": 1,
            },
        },
    },
    "BLADDER": {
        "default_model": "unet",
        "models": {
            "unet": {
                "task": "segmentation",
                "arch": "smp_unet",
                "expected_path": "models/bladder_seg.pt",
                "label_type": "mask_binary",
                "num_classes": 3,
                "input_size": (224, 224),
                "channels": 1,
            },
        },
    },
    "LUNG": {
        "default_model": "classifier",
        "models": {
            "classifier": {
                "task": "classification",
                "arch": "efficientnet",
                "expected_path": "models/lung_classifier.pt",
                "label_type": "class_index",
                "num_classes": 4,
                "input_size": (256, 256),
                "channels": 1,
            },
        },
    },
}


def get_plugin_spec(plugin: str) -> Optional[Dict]:
    return PLUGIN_SPECS.get(plugin.upper())


def get_model_spec(plugin: str, model_key: Optional[str] = None) -> Optional[Dict]:
    plugin_spec = get_plugin_spec(plugin)
    if not plugin_spec:
        return None
    key = model_key or plugin_spec.get("default_model")
    return plugin_spec["models"].get(key)


def get_model_path(plugin: str, model_key: Optional[str] = None) -> Optional[str]:
    spec = get_model_spec(plugin, model_key)
    if not spec:
        return None
    return spec.get("expected_path")
