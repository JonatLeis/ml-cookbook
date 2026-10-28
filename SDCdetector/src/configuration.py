"""Loads and validates configuration files."""
from dataclasses import asdict, fields, replace
import importlib
import json
import math
from pathlib import Path
import re
from typing import get_args, get_origin, get_type_hints


RUN_KEYS = {"seed", "threads", "repeats", "job_timeout"}
CONTROLLER_KEYS = RUN_KEYS | {"logging", "output", "nproc_per_node", "device", "api_host", "api_port", "api_linger"}


def mapping(value, where):
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{where}: expected a mapping with string keys")
    return value


def known(value, allowed, where):
    mapping(value, where)
    unknown = value.keys() - allowed
    if unknown:
        raise ValueError(f"{where}: unknown settings {sorted(unknown)}; allowed: {sorted(allowed)}")
    return value


def environment(value, where):
    mapping(value, where)
    for key, setting in value.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key.startswith("EXPERIMENT_"):
            raise ValueError(f"{where}: invalid or reserved environment name {key}")
        if not isinstance(setting, (str, int, float, bool)) or setting is None:
            raise ValueError(f"{where}.{key}: expected a scalar")
    return {key: str(int(value)) if isinstance(value, bool) else str(value) for key, value in value.items()}


def validate_run(settings, where):
    for key, value in settings.items():
        if key in ("seed", "threads", "repeats", "nproc_per_node", "api_port"):
            if type(value) is not int:
                raise ValueError(f"{where}.{key}: expected integer")
            if key == "seed" and not 0 <= value < 2**63:
                raise ValueError(f"{where}.seed: must be in [0, 2**63)")
            if key in ("threads", "repeats", "nproc_per_node") and value < 1:
                raise ValueError(f"{where}.{key}: must be positive")
            if key == "api_port" and not 0 <= value <= 65535:
                raise ValueError(f"{where}.api_port: invalid port")
        elif key in ("job_timeout", "api_linger"):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{where}.{key}: must be finite and nonnegative")
        elif key == "logging" and value not in ("zero", "minimum", "verbose"):
            raise ValueError(f"{where}.logging: expected zero, minimum or verbose")
        elif key == "device" and value not in ("cpu", "cuda"):
            raise ValueError(f"{where}.device: expected cpu or cuda")
        elif key in ("api_host", "output") and (not isinstance(value, str) or not value):
            raise ValueError(f"{where}.{key}: expected a nonempty string")


def load_config(path):
    import yaml

    # Reject duplicate keys instead of silently losing experiment settings.
    class Loader(yaml.SafeLoader):
        pass

    def construct(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ValueError(f"config: invalid or duplicate key {key!r}")
            result[key] = loader.construct_object(value_node)
        return result

    Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct)
    try:
        data = yaml.load(Path(path).read_text(), Loader=Loader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    known(data, {"version", "controller", "environment", "experiments"}, "config")
    if type(data.get("version")) is not int or data["version"] != 1:
        raise ValueError("config.version must be 1")
    controller = known(data.get("controller", {}), CONTROLLER_KEYS, "controller")
    validate_run(controller, "controller")
    controller = dict(controller)
    if "output" in controller:
        output = Path(controller["output"])
        controller["output"] = output if output.is_absolute() else Path(path).resolve().parent / output
    jobs = mapping(data.get("experiments"), "experiments")
    normalized = {}
    for name, raw in jobs.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"invalid experiment name {name!r}")
        entry = known(raw, {"enabled", "run", "settings", "environment"}, name)
        if type(entry.get("enabled", True)) is not bool:
            raise ValueError(f"{name}.enabled: expected boolean")
        run = known(entry.get("run", {}), RUN_KEYS, name + ".run")
        validate_run(run, name + ".run")
        normalized[name] = {"enabled": entry.get("enabled", True), "run": run,
                            "settings": mapping(entry.get("settings", {}), name + ".settings"),
                            "environment": environment(entry.get("environment", {}), name + ".environment")}
    return {"controller": controller, "experiments": normalized,
            "environment": environment(data.get("environment", {}), "environment")}


def convert(value, annotation, where):
    import torch

    origin, arguments = get_origin(annotation), get_args(annotation)
    if annotation is torch.dtype:
        dtype = getattr(torch, value.removeprefix("torch."), None) if isinstance(value, str) else None
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"{where}: expected a torch dtype name, got {value!r}")
        return dtype
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise ValueError(f"{where}: expected a YAML list")
        variable = origin is list or (len(arguments) == 2 and arguments[1] is Ellipsis)
        if not variable and len(value) != len(arguments):
            raise ValueError(f"{where}: expected {len(arguments)} elements")
        converted = [convert(item, arguments[0] if variable else arguments[index], f"{where}[{index}]")
                     for index, item in enumerate(value)]
        return converted if origin is list else tuple(converted)
    if annotation is float and type(value) in (int, float) and math.isfinite(value):
        return float(value)
    if annotation in (int, str, bool) and type(value) is annotation:
        return value
    raise ValueError(f"{where}: expected {annotation}, got {value!r}")


def configure_experiment(experiment, settings):
    known(settings, {field.name for field in fields(experiment) if field.init}, type(experiment).__name__)
    hints = get_type_hints(type(experiment))
    return replace(experiment, **{key: convert(value, hints[key], key) for key, value in settings.items()})


def load_experiment(name, settings):
    from experiment import Experiment
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("invalid experiment module name")
    experiment = importlib.import_module(f"experiments.{name}").EXPERIMENT
    if not isinstance(experiment, Experiment):
        raise ValueError(f"{name}.EXPERIMENT must be an Experiment instance")
    return configure_experiment(experiment, settings)


def describe(name):
    experiment = load_experiment(name, {})
    defaults = json.loads(json.dumps(asdict(experiment), default=str))
    hints = get_type_hints(type(experiment))
    return {key: {"type": str(hints[key]), "default": value} for key, value in defaults.items()}
