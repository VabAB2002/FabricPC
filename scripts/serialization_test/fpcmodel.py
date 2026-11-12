"""Test code to save and load a FabricPC model (graph structure + weights).

Each model gets its own folder:
    <model>.fpc/
        structure.json   nodes, edges and settings of the graph
        params/          the weights, saved with Orbax
                         (or weights.npz if you set FPC_BACKEND=npz)

To load, we call the normal FabricPC constructors and graph() again,
so you don't need the Python file that first built the model.
This is only a test. The real version should go inside FabricPC
(fabricpc/serialization.py, like in Matthew's checkpointing plan).
"""

import dataclasses
import importlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import jax.numpy as jnp

FORMAT = "fabricpc-model"
FORMAT_VERSION = 1
ALLOWED_PREFIX = "fabricpc."

BACKEND = os.environ.get(
    "FPC_BACKEND", "orbax"
)  # "orbax" saves params/, "npz" saves weights.npz
SEP = "|"  # keys look like node|kind|name, node and edge names never have "|" in them


# ---------------------------------------------------------------- encoding
def _class_path(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_component(obj):
    return hasattr(obj, "config") and not isinstance(obj, (dict, type))


def encode(obj):
    """Turn a value into something JSON can store.

    Tuples and FabricPC classes get a tag, so loading gives back the same thing.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(v) for v in obj]}
    if isinstance(obj, list):
        return [encode(v) for v in obj]
    if _is_component(obj):
        return {
            "__class__": _class_path(type(obj)),
            "config": {k: encode(v) for k, v in dict(obj.config).items()},
        }
    if hasattr(obj, "keys"):  # dict or MappingProxyType
        return {"__dict__": {k: encode(v) for k, v in dict(obj).items()}}
    if isinstance(obj, (np.generic,)):
        return obj.item()
    raise TypeError(f"cannot encode {type(obj).__name__}: {obj!r}")


def import_class(path, base=None, extra_classes=None):
    if extra_classes and path in extra_classes:
        return extra_classes[path]
    if not path.startswith(ALLOWED_PREFIX):
        raise ValueError(
            f"refusing to import {path!r}: only {ALLOWED_PREFIX}* classes load "
            f"by default; pass extra_classes={{path: cls}} for custom ones"
        )
    module_name, _, name = path.rpartition(".")
    cls = getattr(importlib.import_module(module_name), name)
    base = base or _component_bases()
    if (
        not isinstance(cls, type)
        or not cls.__module__.startswith(
            ALLOWED_PREFIX
        )  # also stops non-fabricpc classes that are just imported inside fabricpc
        or not issubclass(cls, base)
    ):
        names = getattr(base, "__name__", "FabricPC component")
        raise ValueError(f"{path!r} is not a fabricpc {names} class")
    return cls


def _component_bases():
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase
    from fabricpc.core.inference import InferenceBase
    from fabricpc.graph_initialization.state_initializer import StateInitBase

    return (
        ActivationBase,
        EnergyFunctional,
        InitializerBase,
        InferenceBase,
        StateInitBase,
    )


# These two classes can't be rebuilt from their own .config, so we build them by hand.
def _ctor_uniform(cls, cfg):
    return cls(min_val=cfg["min"], max_val=cfg["max"])


def _ctor_schedule(cls, cfg):
    return cls(*cfg["solvers"])


CTOR_ADAPTERS = {
    "fabricpc.core.initializers.UniformInitializer": _ctor_uniform,
    "fabricpc.core.inference.InferenceSchedule": _ctor_schedule,
}


def decode(spec, extra_classes=None):
    if isinstance(spec, list):
        return [decode(v, extra_classes) for v in spec]
    if not isinstance(spec, dict):
        return spec
    if "__tuple__" in spec:
        return tuple(decode(v, extra_classes) for v in spec["__tuple__"])
    if "__dict__" in spec:
        return {k: decode(v, extra_classes) for k, v in spec["__dict__"].items()}
    if "__class__" in spec:
        path = spec["__class__"]
        cls = import_class(path, extra_classes=extra_classes)
        cfg = {k: decode(v, extra_classes) for k, v in spec["config"].items()}
        obj = CTOR_ADAPTERS[path](cls, cfg) if path in CTOR_ADAPTERS else cls(**cfg)
        if encode(obj) != spec:  # make sure we rebuilt the exact same thing
            raise ValueError(f"{path} does not round-trip its config: {spec}")
        return obj
    raise ValueError(f"unknown spec {spec!r}")


# ---------------------------------------------------------------- structure
def structure_to_dict(structure):
    import fabricpc

    nodes = []
    for (
        name,
        node,
    ) in structure.nodes.items():  # dict keeps the order the nodes were added
        ni = node.node_info
        sc = ni.scaling_config
        nodes.append(
            {
                "name": name,
                "class": _class_path(ni.node_class),
                "shape": list(ni.shape),
                "activation": encode(ni.activation),
                "energy": encode(ni.energy),
                "latent_init": encode(ni.latent_init),
                "weight_init": encode(ni.weight_init),
                "config": {k: encode(v) for k, v in dict(ni.node_config).items()},
                "mupc": None if sc is None else dataclasses.asdict(sc),
            }
        )
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "fabricpc_version": fabricpc.__version__,
        "nodes": nodes,
        "edges": [
            {"source": e.source, "target": e.target, "slot": e.slot}
            for e in structure.edges.values()
        ],
        "task_map": dict(structure.task_map),
        "node_order": list(structure.node_order),
        "schedule": list(structure.schedule),
        "unroll": structure.config.get("unroll"),
        "inference": encode(structure.config["inference"]),
        "graph_state_initializer": encode(structure.config["graph_state_initializer"]),
    }


def _build_node(nd, extra_classes):
    from fabricpc.nodes.base import NodeBase

    cls = import_class(nd["class"], base=NodeBase, extra_classes=extra_classes)
    kwargs = {
        "shape": tuple(nd["shape"]),
        "name": nd["name"],
        "activation": decode(nd["activation"], extra_classes),
        "energy": decode(nd["energy"], extra_classes),
        "latent_init": decode(nd["latent_init"], extra_classes),
        "weight_init": decode(nd["weight_init"], extra_classes),
    }
    kwargs.update({k: decode(v, extra_classes) for k, v in nd["config"].items()})
    sig = inspect.signature(cls.__init__)
    accepts_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    dropped = {}
    if not accepts_var_kw:
        for k in list(kwargs):
            if k not in sig.parameters:
                dropped[k] = kwargs.pop(k)
    node = cls(**kwargs)
    return node, dropped


def structure_from_dict(d, extra_classes=None, report=None):
    from fabricpc.core.topology import Edge, _get_current_namespace
    from fabricpc.graph_assembly import graph

    if d.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file (format={d.get('format')!r})")
    if d["format_version"] > FORMAT_VERSION:
        raise ValueError(
            f"file is format_version {d['format_version']}; this loader reads "
            f"up to {FORMAT_VERSION}. Upgrade fabricpc."
        )
    if _get_current_namespace():
        raise RuntimeError("load a model outside any GraphNamespace")

    nodes = {}
    for nd in d["nodes"]:
        node, dropped = _build_node(nd, extra_classes)
        if report is not None and dropped:
            report.setdefault("dropped_ctor_kwargs", {})[nd["name"]] = {
                k: repr(v) for k, v in dropped.items()
            }
        nodes[nd["name"]] = node
    edges = [
        Edge(source=nodes[e["source"]], target=nodes[e["target"]].slot(e["slot"]))
        for e in d["edges"]
    ]
    structure = graph(
        nodes=list(nodes.values()),
        edges=edges,
        task_map=dict(d["task_map"]),
        inference=decode(d["inference"], extra_classes),
        graph_state_initializer=decode(d["graph_state_initializer"], extra_classes),
        unroll=d["unroll"],
    )
    # graph() works out the order again, it should match the saved one.
    if list(structure.node_order) != d["node_order"]:
        raise ValueError("rebuilt node_order differs from the saved one")
    if list(structure.schedule) != d["schedule"]:
        raise ValueError("rebuilt schedule differs from the saved one")

    # For muPC we save the scaling numbers directly instead of working them out
    # again (MuPCConfig is not kept in the structure, and this way old files
    # still load the same even if the formula changes later).
    from fabricpc.core.mupc import MuPCScalingFactors

    new_nodes = {}
    for nd in d["nodes"]:
        node = structure.nodes[nd["name"]]
        if nd["mupc"] is not None:
            info = dataclasses.replace(
                node.node_info, scaling_config=MuPCScalingFactors(**nd["mupc"])
            )
            node = node._with_graph_info(info)
        new_nodes[nd["name"]] = node
    return structure._replace(nodes=new_nodes)


# ---------------------------------------------------------------- weights
def params_to_flat(params):
    flat = {}
    for node, np_ in params.nodes.items():
        for kind in ("weights", "biases"):
            for name, arr in getattr(np_, kind).items():
                for part in (node, name):
                    assert SEP not in part, part
                flat[f"{node}{SEP}{kind}{SEP}{name}"] = np.asarray(arr)
    return flat


def params_from_flat(flat, structure):
    from fabricpc.core.types import GraphParams, NodeParams

    nodes = {name: {"weights": {}, "biases": {}} for name in structure.nodes}
    for key, arr in flat.items():
        node, kind, name = key.split(SEP)
        nodes[node][kind][name] = jnp.asarray(arr)
    return GraphParams(
        nodes={n: NodeParams(v["weights"], v["biases"]) for n, v in nodes.items()}
    )


# ---------------------------------------------------------------- public API
def save_model(structure, params, path, metadata=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    d = structure_to_dict(structure)
    flat = params_to_flat(params)
    d["weights"] = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in flat.items()
    }
    d["metadata"] = metadata or {}
    (path / "structure.json").write_text(json.dumps(d, indent=1))
    if BACKEND == "npz":
        np.savez(path / "weights.npz", **flat)
    else:
        import orbax.checkpoint as ocp

        # Save a normal nested dict (not GraphParams), so Orbax can restore it without a template.
        plain = {
            n: {"weights": dict(v.weights), "biases": dict(v.biases)}
            for n, v in params.nodes.items()
        }
        ck = ocp.StandardCheckpointer()
        ck.save((path / "params").resolve(), plain)
        ck.wait_until_finished()
    return path


def load_model(path, extra_classes=None, report=None):
    path = Path(path)
    d = json.loads((path / "structure.json").read_text())
    structure = structure_from_dict(d, extra_classes, report)
    # Quick check: the loaded structure should give the same json as the file.
    again = json.loads(json.dumps(structure_to_dict(structure)))
    for k in (
        "nodes",
        "edges",
        "task_map",
        "node_order",
        "schedule",
        "unroll",
        "inference",
        "graph_state_initializer",
    ):
        if again[k] != d[k]:
            raise ValueError(f"rebuilt structure differs from structure.json in {k!r}")
    if (path / "weights.npz").exists():
        with np.load(path / "weights.npz", allow_pickle=False) as z:
            flat = {k: z[k] for k in z.files}
    else:
        import orbax.checkpoint as ocp

        tree = ocp.StandardCheckpointer().restore((path / "params").resolve())
        flat = {
            f"{n}{SEP}{kind}{SEP}{name}": np.asarray(arr)
            for n, v in tree.items()
            for kind in ("weights", "biases")
            for name, arr in v.get(kind, {}).items()
        }
    for k, meta in d["weights"].items():
        a = flat[k]
        if list(a.shape) != meta["shape"] or str(a.dtype) != meta["dtype"]:
            raise ValueError(f"weight {k} does not match structure.json")
    if set(flat) != set(d["weights"]):
        raise ValueError("weights.npz keys differ from structure.json")
    return structure, params_from_flat(flat, structure), d["metadata"]
