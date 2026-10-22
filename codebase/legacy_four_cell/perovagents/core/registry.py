"""Tool registry + tool knowledge graph (Orchestration layer).

The registry is the agent's address book of capabilities. Tools self-register
by declaring `name`, `layer`, and `capabilities`. The layer exposes:

  * `ToolRegistry.available()`      — inventory of tools (for the planner prompt)
  * `ToolRegistry.get(name)`        — instantiate a tool by name
  * `ToolRegistry.validate_action/validate_params` — plan-level schema gate
  * `ToolKnowledgeGraph.related()`  — which tools commonly work together

Tools are *lazily instantiated*: `register()` stores a factory, the instance is
created on first `get()`. This keeps import cost low and lets a planner see the
whole catalog without loading heavy models.

Action schemas
--------------
Every tool declares a set of *actions* (``register(..., actions=...)``), each
with its own input schema, output contract, task types it satisfies, and
downstream compatibility:

    tool ── action 1
        │       ├── input schema  (params + types + enums)
        │       ├── output tokens (what the result will carry)
        │       ├── preconditions / requires
        │       └── produces / compatible_with
        ├── action 2 ...

This is what the planner plans against: an action may only be selected if it
supports the required task type, and its params are validated *before*
execution. Unvalidated plans are rejected, never silently coerced.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .tool import Tool


@dataclass
class ActionSpec:
    """Declarative schema for ONE tool action (input/output/preconditions)."""
    name: str
    description: str = ""
    task_types: Set[str] = field(default_factory=set)
    params: Dict[str, dict] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    compatible_with: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "task_types": sorted(self.task_types),
            "params": self.params,
            "outputs": list(self.outputs),
            "produces": list(self.produces),
            "compatible_with": list(self.compatible_with),
            "requires": list(self.requires),
            "constraints": list(self.constraints),
        }


@dataclass
class ToolSpec:
    """Declarative description of a tool before it is instantiated."""
    name: str
    factory: Callable[[], Tool]
    description: str = ""
    layer: str = "core"
    capabilities: Set[str] = field(default_factory=set)
    version: str = "0.1.0"
    actions: Dict[str, ActionSpec] = field(default_factory=dict)

    def to_dict(self):
        d = {"name": self.name, "description": self.description,
             "layer": self.layer, "version": self.version,
             "capabilities": sorted(self.capabilities)}
        if self.actions:
            d["actions"] = [a.to_dict() for a in self.actions.values()]
        return d

    def action_spec(self, name: str) -> Optional[ActionSpec]:
        return self.actions.get(name)


class ToolRegistry:
    """Stores ToolSpecs, lazily materializes Tool instances."""

    def __init__(self):
        self._specs: Dict[str, ToolSpec] = {}
        self._instances: Dict[str, Tool] = {}

    # ---- registration ----------------------------------------------------
    def register(self, tool_cls: type, factory: Optional[Callable[[], Tool]] = None,
                 actions: Optional[dict] = None, parameters: Optional[dict] = None):
        """Register a Tool *class*. Instances are created on demand.

        `actions` optionally documents per-action schemas for the LLM planner:
            {action_name: ActionSpec | str}   (a bare str becomes the
            description). `parameters` is the legacy flat schema; retained for
            backwards compatibility and merged into the 'query'/'call' action
            shape when the tool declares no explicit actions.
        """
        if actions:
            specs = {}
            for name, a in actions.items():
                if isinstance(a, ActionSpec):
                    a = dataclasses_replace(a, name=name) if a.name != name else a
                    specs[name] = a
                elif isinstance(a, str):
                    specs[name] = ActionSpec(name=name, description=a)
                elif isinstance(a, dict):
                    specs[name] = ActionSpec(**{**a, "name": name})
            actions = specs
        elif parameters:
            actions = {"query": ActionSpec(
                name="query", description="default action for this tool",
                params=parameters, outputs=["result"])}
        else:
            actions = {}
        spec = ToolSpec(
            name=tool_cls.name,
            factory=factory or tool_cls,
            description=getattr(tool_cls, "description", ""),
            layer=getattr(tool_cls, "layer", "core"),
            capabilities=set(getattr(tool_cls, "capabilities", set())),
            version=getattr(tool_cls, "version", "0.1.0"),
            actions=actions,
        )
        self._specs[spec.name] = spec
        self._instances.pop(spec.name, None)
        return spec

    def register_instance(self, tool: Tool, actions: Optional[dict] = None):
        """Register an already-constructed tool instance."""
        self._instances[tool.name] = tool
        self._specs[tool.name] = ToolSpec(
            name=tool.name, factory=lambda: tool,
            description=tool.description, layer=tool.layer,
            capabilities=set(tool.capabilities), version=tool.version,
            actions=actions or {})

    # ---- reads -----------------------------------------------------------
    def get(self, name: str) -> Optional[Tool]:
        spec = self._specs.get(name)
        if spec is None:
            return None
        if name not in self._instances:
            try:
                self._instances[name] = spec.factory()
            except Exception:
                return None
        return self._instances[name]

    def available(self) -> List[ToolSpec]:
        return sorted(self._specs.values(), key=lambda s: s.name)

    def names(self) -> List[str]:
        return [s.name for s in self.available()]

    def by_layer(self, layer: str) -> List[ToolSpec]:
        return [s for s in self.available() if s.layer == layer]

    def by_capability(self, cap: str) -> List[ToolSpec]:
        return [s for s in self.available() if cap in s.capabilities]

    def catalog_dicts(self) -> List[Dict]:
        return [s.to_dict() for s in self.available()]

    # ---- plan validation ------------------------------------------------
    def action_spec(self, tool: str, action: str) -> Optional[ActionSpec]:
        spec = self._specs.get(tool)
        if spec is None:
            return None
        if not action:
            specs = list(spec.actions.values())
            return specs[0] if len(specs) == 1 else None
        return spec.actions.get(action)

    def validate_action(self, tool: str, action: str) -> bool:
        """Does `tool` declare `action`? (False for unknown tool/action.)"""
        return self.action_spec(tool, action) is not None

    def action_supports_task(self, tool: str, action: str, task_type: str) -> bool:
        """Can this action satisfy the atomic task type the analyzer assigned?"""
        a = self.action_spec(tool, action)
        return bool(a and task_type in a.task_types)

    def validate_params(self, tool: str, action: str,
                        params: Dict[str, Any]) -> tuple:
        """Type/enum check of `params` against the action's input schema.

        Returns (ok: bool, errors: list[str], warnings: list[str]).
        Errors are hard rejections (bad type, invalid enum, missing required).
        Unknown extra params become warnings (inert, non-conflicting keys are
        tolerated to avoid fragile replan churn), unless the schema declares
        ``"strict": true`` for the action.
        """
        a = self.action_spec(tool, action)
        if a is None:
            return False, [f"unknown action '{action}' for tool '{tool}'"], []
        errors, warnings = [], []
        schema = a.params or {}
        strict = schema.get("_strict") is True
        for p, spec in schema.items():
            if p == "_strict":
                continue
            if p not in params and spec.get("required"):
                errors.append(f"missing required param '{p}' for {tool}.{action}")
        for key, val in (params or {}).items():
            spec = schema.get(key)
            if spec is None:
                if strict or spec.get("_strict"):
                    msg = (f"'{key}' is not a param of {tool}.{a.name}; if you "
                           f"want an explanation, use ml.explain_global or "
                           f"ml.explain_local — do not pass flags to simulate "
                           f"a dedicated action")
                    errors.append(msg)
                else:
                    warnings.append(f"unused param '{key}' ignored for {tool}.{action}")
                continue
            ptype = spec.get("type")
            if not _check_type(val, ptype) and spec.get("required"):
                errors.append(f"param '{key}' of {tool}.{action} must be "
                              f"{ptype}, got {type(val).__name__}")
                continue
            enum = spec.get("enum")
            if enum and val not in enum:
                errors.append(f"param '{key}' of {tool}.{action} must be one of "
                              f"{enum}, got {val!r}")
        return not errors, errors, warnings

    def validate_step(self, tool: str, action: str,
                      params: Dict[str, Any]) -> tuple:
        """Validate action existence and params at once."""
        return self.validate_action(tool, action), *self.validate_params(
            tool, action, params)[1:]


def _check_type(val, ptype: Optional[str]) -> bool:
    if ptype is None or val is None:
        return True
    if ptype == "string":
        return isinstance(val, str)
    if ptype in ("number", "float"):
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if ptype == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if ptype == "boolean":
        return isinstance(val, bool)
    if ptype in ("array", "list"):
        return isinstance(val, (list, tuple))
    if ptype == "object":
        return isinstance(val, dict)
    return True


def dataclasses_replace(a: ActionSpec, **kw) -> ActionSpec:
    """Shallow-copy an ActionSpec with selected fields overridden
    (ActionSpec is frozen, so updates go through copy-on-write)."""
    return ActionSpec(**{**a.__dict__, **kw})


class ToolKnowledgeGraph:
    """Graph over tools built from capabilities + action compatibility.

    Two tools are connected when they share a capability tag or a curated edge
    exists. Action metadata (`produces` / `compatible_with`) also links tools,
    so e.g. ml.explain_global (produces feature_name) is connected to
    ml.what_if (consumes a feature as its `parameter`) and to pdb.query.
    """

    CO_USE = [
        ("ml", "material", "predictor resolves materials to features"),
        ("ml", "pdb", "predictor trained on the experimental DB"),
        ("scaps", "pdb", "physics sensitivity vs empirical trends"),
        ("literature", "ml", "literature rationalizes predictions"),
        ("pdb", "material", "device stats benefit from material context"),
        ("scaps", "literature", "physics vs published explanations"),
    ]

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self._edges: Dict[str, Set[str]] = {}
        self._build()

    def _build(self):
        specs = {s.name: s for s in self.registry.available()}
        caps: Dict[str, Set[str]] = {}
        # capability edges
        for name, spec in specs.items():
            for c in spec.capabilities:
                caps.setdefault(c, set()).add(name)
        for members in caps.values():
            for a in members:
                for b in members:
                    if a != b:
                        self._add(a, b)
        # two-hop action compatibility edges: producer -> consumer
        for name, spec in specs.items():
            for act in spec.actions.values():
                for comp in act.compatible_with:
                    target = comp.split(".")[0].strip()
                    if target in specs:
                        self._add(name, target)
        # curated co-use edges
        for a, b, _ in self.CO_USE:
            if a in specs and b in specs:
                self._add(a, b)

    def _add(self, a: str, b: str):
        self._edges.setdefault(a, set()).add(b)
        self._edges.setdefault(b, set()).add(a)

    def related(self, tool_name: str) -> List[str]:
        return sorted(self._edges.get(tool_name, set()))

    def to_dict(self) -> Dict[str, list]:
        return {k: sorted(v) for k, v in self._edges.items()}


def default_registry() -> ToolRegistry:
    """Build the registry with every tool this framework ships."""
    from .. import layers  # noqa: F401  (registers tools via side-effect)
    registry = ToolRegistry()
    layers.register_all(registry)
    return registry