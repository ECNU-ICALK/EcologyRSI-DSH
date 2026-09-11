"""Declarative, host-compiled feature recipes for candidate forecasting models.

The Agent authors JSON that *names* host primitives; it never supplies code.
Every term is checked against ``RECIPE_PRIMITIVES`` -- the single source of truth
for which operations exist and what bounds their parameters obey -- and then
compiled here into an explicit read plan.

A compiled plan states the timestamp offsets it needs relative to the forecast
origin, so the fit path and the Agent tool inference path resolve the same
history instead of each re-deriving a window from ``history_steps``.  Every
offset must be non-positive: a recipe cannot express a read of the future, and
``seasonal_reference`` is rejected outright when the horizon exceeds its period.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from statistics import fmean, pstdev
from typing import Any, Callable, Mapping, Sequence

from ..core.models import canonical_json, digest

FEATURE_RECIPE_SCHEMA_VERSION = "ecologyrsi-dsh.feature-recipe/1"

MAX_RECIPE_TERMS = 12
MAX_RECIPE_LAG_HOURS = 168
MAX_ROLLING_WINDOW_HOURS = 168
MAX_SLOPE_WINDOW_HOURS = 48
MAX_EXOGENOUS_LAG_HOURS = 24
MAX_EXOGENOUS_ROLLING_WINDOW_HOURS = 48
ALLOWED_SEASONAL_PERIODS = (24, 168)
ALLOWED_MODEL_KINDS = frozenset({"ridge"})
ALLOWED_MODEL_ANCHORS = frozenset({"fit_selected_baseline", "persistence"})
MODEL_ALPHA_MINIMUM = 1e-4
# Matches the registered ``ridge_alpha`` contract; widening it would have to be
# propagated to every registered predictor bound, so the recipe stays aligned.
MODEL_ALPHA_MAXIMUM = 1.0
RESIDUAL_SCALE_MINIMUM = 0.0
RESIDUAL_SCALE_MAXIMUM = 1.0

RECIPE_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "features", "model", "per_horizon", "per_cell"}
)
_MODEL_KEYS = frozenset({"kind", "alpha", "anchor"})
_SCALE_KEYS = frozenset({"residual_scale"})
_COLUMN_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_HORIZON_KEY_RE = re.compile(r"[1-9][0-9]{0,3}")
_CELL_KEY_RE = re.compile(r"([a-z][a-z0-9_]{0,63})@([1-9][0-9]{0,3})h")

# Stands in for a concrete target when only a recipe's structure is being
# projected. Deliberately not a legal column name (`_COLUMN_RE` rejects it), so
# a structural summary can never be mistaken for one cell's real feature names.
STRUCTURAL_TARGET_PLACEHOLDER = "*"


@dataclass(frozen=True, slots=True)
class PrimitiveParameter:
    """One declarative knob of a primitive; bounds are enforced by the host."""

    name: str
    kind: str
    minimum: int | None = None
    maximum: int | None = None
    choices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        # The name is carried by the caller's key, not repeated here: the
        # grammar travels to the sample Agent inside plan.tools[*].parameters,
        # and sample_contracts._safe_value refuses anything nested deeper than
        # ten levels. Every level this projection saves is a level the wire
        # budget does not have to spend.
        body: dict[str, Any] = {"kind": self.kind}
        if self.choices:
            body["choices"] = list(self.choices)
        if self.minimum is not None:
            body["minimum"] = self.minimum
        if self.maximum is not None:
            body["maximum"] = self.maximum
        return body

    def validate(self, value: Any, *, path: str) -> int | str:
        if self.kind == "dataset_feature":
            if not isinstance(value, str) or not _COLUMN_RE.fullmatch(value):
                raise ValueError(f"{path} must be a lowercase dataset feature name")
            return value
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{path} must be an integer")
        if self.choices and value not in self.choices:
            allowed = ", ".join(str(choice) for choice in self.choices)
            raise ValueError(f"{path} must be one of: {allowed}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{path} must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{path} must be <= {self.maximum}")
        return value


@dataclass(frozen=True, slots=True)
class _CompiledTerm:
    feature_names: tuple[str, ...]
    target_offsets: tuple[int, ...]
    exogenous_offsets: tuple[tuple[str, int], ...]
    evaluate: Callable[
        [int, Mapping[int, float], Mapping[tuple[str, int], float]], tuple[float, ...]
    ]


@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
    """A host-owned primitive: declarative bounds plus the host's compiler."""

    op: str
    parameters: tuple[PrimitiveParameter, ...]
    feature_arity: int
    purpose: str
    compile: Callable[..., _CompiledTerm]

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset(parameter.name for parameter in self.parameters)

    def to_dict(self) -> dict[str, Any]:
        # Parameter bounds are published separately, under flat "<op>.<name>"
        # keys, so this entry stays three levels shallower than the nested
        # list-of-objects shape the wire depth fence rejects.
        return {
            "purpose": self.purpose,
            "feature_arity": self.feature_arity,
            "parameters": [parameter.name for parameter in self.parameters],
        }


def _compile_target_lag(term: Mapping[str, Any], *, target: str, horizon: int) -> _CompiledTerm:
    del horizon
    offset = -int(term["k"])
    return _CompiledTerm(
        feature_names=(f"target:{target}:lag_{term['k']}h",),
        target_offsets=(offset,),
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (reads[offset],),
    )


def _compile_seasonal_reference(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    period = int(term["period"])
    if horizon > period:
        raise ValueError(
            "seasonal_reference would read the future: "
            f"horizon {horizon}h exceeds period {period}h"
        )
    offset = horizon - period
    return _CompiledTerm(
        feature_names=(f"target:{target}:seasonal_{period}h_reference",),
        target_offsets=(offset,),
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (reads[offset],),
    )


def _compile_seasonal_delta(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del horizon
    period = int(term["period"])
    return _CompiledTerm(
        feature_names=(f"target:{target}:seasonal_{period}h_delta",),
        target_offsets=(0, -period),
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (reads[0] - reads[-period],),
    )


def _compile_diurnal_sin_cos(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del target, horizon
    period = int(term["period"])

    def evaluate(
        origin: int,
        reads: Mapping[int, float],
        external: Mapping[tuple[str, int], float],
    ) -> tuple[float, ...]:
        del reads, external
        angle = 2.0 * math.pi * (origin % period) / period
        return (math.sin(angle), math.cos(angle))

    return _CompiledTerm(
        feature_names=(f"time:sin_{period}h", f"time:cos_{period}h"),
        target_offsets=(),
        exogenous_offsets=(),
        evaluate=evaluate,
    )


def _compile_rolling_mean(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del horizon
    window = int(term["w"])
    offsets = tuple(-step for step in range(window))
    return _CompiledTerm(
        feature_names=(f"target:{target}:rolling_mean_{window}h",),
        target_offsets=offsets,
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (
            fmean(reads[offset] for offset in offsets),
        ),
    )


def _compile_rolling_std(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del horizon
    window = int(term["w"])
    offsets = tuple(-step for step in range(window))
    return _CompiledTerm(
        feature_names=(f"target:{target}:rolling_std_{window}h",),
        target_offsets=offsets,
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (
            pstdev(reads[offset] for offset in offsets),
        ),
    )


def _compile_target_slope(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del horizon
    window = int(term["w"])
    return _CompiledTerm(
        feature_names=(f"target:{target}:slope_{window}h",),
        target_offsets=(0, -window),
        exogenous_offsets=(),
        evaluate=lambda origin, reads, external: (
            (reads[0] - reads[-window]) / window,
        ),
    )


def _compile_exogenous(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del target, horizon
    column = str(term["col"])
    lag = int(term["lag"])
    key = (column, -lag)
    return _CompiledTerm(
        feature_names=(f"exogenous:{column}:lag_{lag}h",),
        target_offsets=(),
        exogenous_offsets=(key,),
        evaluate=lambda origin, reads, external: (external[key],),
    )


def _compile_exogenous_rolling_mean(
    term: Mapping[str, Any], *, target: str, horizon: int
) -> _CompiledTerm:
    del target, horizon
    column = str(term["col"])
    window = int(term["w"])
    keys = tuple((column, -step) for step in range(window))
    return _CompiledTerm(
        feature_names=(f"exogenous:{column}:rolling_mean_{window}h",),
        target_offsets=(),
        exogenous_offsets=keys,
        evaluate=lambda origin, reads, external: (
            fmean(external[key] for key in keys),
        ),
    )


RECIPE_PRIMITIVES: Mapping[str, PrimitiveSpec] = {
    spec.op: spec
    for spec in (
        PrimitiveSpec(
            op="target_lag",
            parameters=(
                PrimitiveParameter("k", "integer", minimum=0, maximum=MAX_RECIPE_LAG_HOURS),
            ),
            feature_arity=1,
            purpose="Target value observed k hours before the forecast origin.",
            compile=_compile_target_lag,
        ),
        PrimitiveSpec(
            op="seasonal_reference",
            parameters=(
                PrimitiveParameter("period", "integer", choices=ALLOWED_SEASONAL_PERIODS),
            ),
            feature_arity=1,
            purpose=(
                "Target value one full period before the label timestamp; the same "
                "cell the seasonal baseline reads, so the candidate is no longer "
                "blind to information the baseline already uses."
            ),
            compile=_compile_seasonal_reference,
        ),
        PrimitiveSpec(
            op="seasonal_delta",
            parameters=(
                PrimitiveParameter("period", "integer", choices=ALLOWED_SEASONAL_PERIODS),
            ),
            feature_arity=1,
            purpose="Change in the target between the origin and one period earlier.",
            compile=_compile_seasonal_delta,
        ),
        PrimitiveSpec(
            op="diurnal_sin_cos",
            parameters=(
                PrimitiveParameter("period", "integer", choices=ALLOWED_SEASONAL_PERIODS),
            ),
            feature_arity=2,
            purpose="Sine and cosine encoding of the origin's phase in the period.",
            compile=_compile_diurnal_sin_cos,
        ),
        PrimitiveSpec(
            op="rolling_mean",
            parameters=(
                PrimitiveParameter("w", "integer", minimum=2, maximum=MAX_ROLLING_WINDOW_HOURS),
            ),
            feature_arity=1,
            purpose="Mean of the target over the w hours ending at the origin.",
            compile=_compile_rolling_mean,
        ),
        PrimitiveSpec(
            op="rolling_std",
            parameters=(
                PrimitiveParameter("w", "integer", minimum=2, maximum=MAX_ROLLING_WINDOW_HOURS),
            ),
            feature_arity=1,
            purpose="Population standard deviation of the target over w hours.",
            compile=_compile_rolling_std,
        ),
        PrimitiveSpec(
            op="target_slope",
            parameters=(
                PrimitiveParameter("w", "integer", minimum=2, maximum=MAX_SLOPE_WINDOW_HOURS),
            ),
            feature_arity=1,
            purpose="Average hourly change of the target across the last w hours.",
            compile=_compile_target_slope,
        ),
        PrimitiveSpec(
            op="exogenous",
            parameters=(
                PrimitiveParameter("col", "dataset_feature"),
                PrimitiveParameter("lag", "integer", minimum=0, maximum=MAX_EXOGENOUS_LAG_HOURS),
            ),
            feature_arity=1,
            purpose="Causally forward-filled exogenous column at a bounded lag.",
            compile=_compile_exogenous,
        ),
        PrimitiveSpec(
            op="exogenous_rolling_mean",
            parameters=(
                PrimitiveParameter("col", "dataset_feature"),
                PrimitiveParameter(
                    "w", "integer", minimum=2, maximum=MAX_EXOGENOUS_ROLLING_WINDOW_HOURS
                ),
            ),
            feature_arity=1,
            purpose="Mean of an exogenous column over the w hours ending at the origin.",
            compile=_compile_exogenous_rolling_mean,
        ),
    )
}


@dataclass(frozen=True, slots=True)
class FrozenRecipe:
    """A validated recipe: plain data, already normalized and content addressed."""

    features: tuple[Mapping[str, Any], ...]
    model: Mapping[str, Any]
    per_horizon: Mapping[str, Mapping[str, Any]]
    per_cell: Mapping[str, Mapping[str, Any]]

    @property
    def term_count(self) -> int:
        return len(self.features)

    @property
    def feature_arity(self) -> int:
        return sum(RECIPE_PRIMITIVES[str(term["op"])].feature_arity for term in self.features)

    @property
    def ridge_alpha(self) -> float:
        return float(self.model["alpha"])

    @property
    def anchor(self) -> str:
        return str(self.model["anchor"])

    @property
    def referenced_columns(self) -> tuple[str, ...]:
        return tuple(
            sorted({str(term["col"]) for term in self.features if "col" in term})
        )

    @property
    def horizons(self) -> tuple[int, ...]:
        """Every horizon the recipe declares a residual scale for.

        The union of both scale maps, not just ``per_horizon``: a recipe is
        allowed to carry only per-cell scales, and a summary that missed those
        horizons would understate the history the recipe demands.
        """

        horizons = {int(key) for key in self.per_horizon}
        for key in self.per_cell:
            matched = _CELL_KEY_RE.fullmatch(key)
            if matched is not None:
                horizons.add(int(matched.group(2)))
        return tuple(sorted(horizons))

    def residual_scale_for(self, target: str, horizon_hours: int) -> float:
        """Per-cell override wins over per-horizon; absent means no residual."""

        cell = self.per_cell.get(f"{target}@{horizon_hours}h")
        if cell is not None:
            return float(cell["residual_scale"])
        horizon = self.per_horizon.get(str(horizon_hours))
        if horizon is not None:
            return float(horizon["residual_scale"])
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": FEATURE_RECIPE_SCHEMA_VERSION,
            "features": [dict(term) for term in self.features],
            "model": dict(self.model),
        }
        if self.per_horizon:
            body["per_horizon"] = {
                key: dict(value) for key, value in self.per_horizon.items()
            }
        if self.per_cell:
            body["per_cell"] = {key: dict(value) for key, value in self.per_cell.items()}
        return body

    @property
    def digest(self) -> str:
        return digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class FeaturePlan:
    """A recipe compiled for exactly one (target, horizon) cell."""

    target: str
    horizon_hours: int
    feature_names: tuple[str, ...]
    required_timestamp_offsets: tuple[int, ...]
    exogenous_timestamp_offsets: Mapping[str, tuple[int, ...]]
    recipe_digest: str
    _terms: tuple[_CompiledTerm, ...]

    @property
    def max_history_hours(self) -> int:
        offsets = [*self.required_timestamp_offsets]
        offsets.extend(
            offset
            for offsets_for_column in self.exogenous_timestamp_offsets.values()
            for offset in offsets_for_column
        )
        return -min(offsets) if offsets else 0

    @property
    def exogenous_columns(self) -> tuple[str, ...]:
        return tuple(sorted(self.exogenous_timestamp_offsets))

    def row(
        self,
        *,
        origin_timestamp: int,
        target_reads: Mapping[int, float],
        exogenous_reads: Mapping[tuple[str, int], float],
    ) -> tuple[float, ...]:
        """Build the feature row from already-resolved reads keyed by offset."""

        values: list[float] = []
        for term in self._terms:
            values.extend(term.evaluate(origin_timestamp, target_reads, exogenous_reads))
        if len(values) != len(self.feature_names):
            raise ArithmeticError("compiled recipe produced an unexpected feature width")
        if not all(math.isfinite(value) for value in values):
            raise ArithmeticError(
                f"non-finite recipe feature for {self.target}@{self.horizon_hours}h"
            )
        return tuple(values)

    def summary(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "horizon_hours": self.horizon_hours,
            "feature_names": list(self.feature_names),
            "required_timestamp_offsets": list(self.required_timestamp_offsets),
            "exogenous_timestamp_offsets": {
                column: list(offsets)
                for column, offsets in sorted(self.exogenous_timestamp_offsets.items())
            },
            "max_history_hours": self.max_history_hours,
            "recipe_digest": self.recipe_digest,
        }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must use string keys")
    return value


def _exact_keys(value: Mapping[str, Any], name: str, allowed: frozenset[str]) -> None:
    unknown = set(value) - allowed
    missing = allowed - set(value)
    if unknown:
        raise ValueError(f"{name} has unsupported fields: {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(sorted(missing))}")


def _bounded_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return number


def _normalized_term(value: Any, *, index: int) -> dict[str, Any]:
    term = _mapping(value, f"features[{index}]")
    op = term.get("op")
    if not isinstance(op, str) or op not in RECIPE_PRIMITIVES:
        allowed = ", ".join(sorted(RECIPE_PRIMITIVES))
        raise ValueError(f"features[{index}].op must be one of: {allowed}")
    spec = RECIPE_PRIMITIVES[op]
    _exact_keys(term, f"features[{index}]", spec.parameter_names | {"op"})
    normalized: dict[str, Any] = {"op": op}
    for parameter in spec.parameters:
        normalized[parameter.name] = parameter.validate(
            term[parameter.name], path=f"features[{index}].{parameter.name}"
        )
    return normalized


def _normalized_scales(
    value: Any, name: str, key_pattern: re.Pattern[str]
) -> dict[str, dict[str, Any]]:
    mapping = _mapping(value, name)
    normalized: dict[str, dict[str, Any]] = {}
    for key in sorted(mapping):
        if not key_pattern.fullmatch(key):
            raise ValueError(f"{name} has an unsupported key: {key}")
        entry = _mapping(mapping[key], f"{name}.{key}")
        _exact_keys(entry, f"{name}.{key}", _SCALE_KEYS)
        normalized[key] = {
            "residual_scale": _bounded_number(
                entry["residual_scale"],
                f"{name}.{key}.residual_scale",
                RESIDUAL_SCALE_MINIMUM,
                RESIDUAL_SCALE_MAXIMUM,
            )
        }
    return normalized


def validate_feature_recipe(
    value: Any, *, allowed_columns: Sequence[str] | None = None
) -> FrozenRecipe:
    """Validate an Agent-authored recipe against the host primitive whitelist.

    Structural validation only: column names are checked for shape here and
    against the actual dataset in :func:`compile_feature_plan`, because genome
    validation runs without a loaded series.
    """

    recipe = _mapping(value, "feature_recipe")
    unknown = set(recipe) - RECIPE_TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"feature_recipe has unsupported fields: {', '.join(sorted(unknown))}"
        )
    declared = recipe.get("schema_version", FEATURE_RECIPE_SCHEMA_VERSION)
    if declared != FEATURE_RECIPE_SCHEMA_VERSION:
        raise ValueError("unsupported feature_recipe schema version")

    features = recipe.get("features")
    if not isinstance(features, Sequence) or isinstance(features, (str, bytes)):
        raise TypeError("feature_recipe.features must be a list")
    if not 1 <= len(features) <= MAX_RECIPE_TERMS:
        raise ValueError(f"feature_recipe.features must hold 1..{MAX_RECIPE_TERMS} terms")
    normalized_features: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, term in enumerate(features):
        normalized = _normalized_term(term, index=index)
        key = canonical_json(normalized)
        if key in seen:
            raise ValueError(f"feature_recipe.features[{index}] duplicates an earlier term")
        seen.add(key)
        normalized_features.append(normalized)

    model = _mapping(recipe.get("model"), "feature_recipe.model")
    _exact_keys(model, "feature_recipe.model", _MODEL_KEYS)
    kind = model["kind"]
    if kind not in ALLOWED_MODEL_KINDS:
        raise ValueError(f"unsupported feature_recipe model kind: {kind}")
    anchor = model["anchor"]
    if anchor not in ALLOWED_MODEL_ANCHORS:
        raise ValueError(f"unsupported feature_recipe model anchor: {anchor}")
    normalized_model = {
        "kind": str(kind),
        "alpha": _bounded_number(
            model["alpha"],
            "feature_recipe.model.alpha",
            MODEL_ALPHA_MINIMUM,
            MODEL_ALPHA_MAXIMUM,
        ),
        "anchor": str(anchor),
    }

    per_horizon = _normalized_scales(
        recipe.get("per_horizon", {}), "feature_recipe.per_horizon", _HORIZON_KEY_RE
    )
    per_cell = _normalized_scales(
        recipe.get("per_cell", {}), "feature_recipe.per_cell", _CELL_KEY_RE
    )

    if allowed_columns is not None:
        permitted = set(allowed_columns)
        unavailable = sorted(
            {str(term["col"]) for term in normalized_features if "col" in term} - permitted
        )
        if unavailable:
            raise ValueError(
                f"feature_recipe references unavailable columns: {', '.join(unavailable)}"
            )

    frozen = FrozenRecipe(
        features=tuple(normalized_features),
        model=normalized_model,
        per_horizon=per_horizon,
        per_cell=per_cell,
    )
    _reject_executable_fields(frozen.to_dict())
    return frozen


def _reject_executable_fields(body: Mapping[str, Any]) -> None:
    """Defence in depth: the key whitelist above already makes this unreachable.

    Imported lazily because ``ecologyrsi_dsh.knowledge`` pulls in the evaluators
    package, and importing it at module scope would close an import cycle.
    """

    from ..knowledge.autonomous_cycle import reject_executable_fields

    reject_executable_fields(body, path="$.feature_recipe")


def recipe_digest(recipe: FrozenRecipe | Mapping[str, Any]) -> str:
    if isinstance(recipe, FrozenRecipe):
        return recipe.digest
    return validate_feature_recipe(recipe).digest


def compile_feature_plan(
    recipe: FrozenRecipe | Mapping[str, Any],
    *,
    series: Any = None,
    target: str,
    horizon_hours: int,
    allowed_roles: frozenset[str] = frozenset(),
    max_history_hours: int | None = None,
) -> FeaturePlan:
    """Compile one cell's read plan, refusing any non-causal offset.

    ``max_history_hours`` is the run's frozen cohort alignment. A recipe that
    reaches further back than the cohort guarantees would be scored on fewer
    origins than its rivals, so it is rejected rather than silently narrowed.

    ``series`` may be omitted to compile the *structure* alone — every offset,
    arity and causality check still runs, only the two checks that need real
    data (that the target exists, and that an exogenous column carries an
    admissible role) are skipped. The behavior compiler needs exactly that: it
    must stay instance-free, yet it has to record how much history the recipe
    will demand before any dataset is opened.
    """

    frozen = recipe if isinstance(recipe, FrozenRecipe) else validate_feature_recipe(recipe)
    if isinstance(horizon_hours, bool) or not isinstance(horizon_hours, int) or horizon_hours < 1:
        raise ValueError("recipe horizon must be a positive whole number of hours")
    if series is not None and target not in getattr(series, "values", {}):
        raise ValueError(f"recipe target is not present in the dataset: {target}")

    terms: list[_CompiledTerm] = []
    feature_names: list[str] = []
    # The anchor value y(origin) is always read: it is the persistence baseline
    # and the row is dropped without it, exactly as the fixed-window path does.
    target_offsets: set[int] = {0}
    exogenous_offsets: dict[str, set[int]] = {}
    for index, term in enumerate(frozen.features):
        spec = RECIPE_PRIMITIVES[str(term["op"])]
        column = term.get("col")
        if column is not None and series is not None:
            specification = series.features.get(column)
            role = getattr(specification, "role", None)
            if column not in series.values or role not in allowed_roles:
                raise ValueError(
                    f"features[{index}].col is not an admissible exogenous column: {column}"
                )
        compiled = spec.compile(term, target=target, horizon=horizon_hours)
        if any(offset > 0 for offset in compiled.target_offsets):
            raise ValueError(f"features[{index}] would read the target after the origin")
        if any(offset > 0 for _, offset in compiled.exogenous_offsets):
            raise ValueError(f"features[{index}] would read an exogenous value after the origin")
        if len(compiled.feature_names) != spec.feature_arity:
            raise ArithmeticError(f"primitive {spec.op} violated its declared arity")
        terms.append(compiled)
        feature_names.extend(compiled.feature_names)
        target_offsets.update(compiled.target_offsets)
        for name, offset in compiled.exogenous_offsets:
            exogenous_offsets.setdefault(name, set()).add(offset)

    if len(set(feature_names)) != len(feature_names):
        raise ValueError("compiled recipe produced duplicate feature names")
    plan = FeaturePlan(
        target=target,
        horizon_hours=horizon_hours,
        feature_names=tuple(feature_names),
        required_timestamp_offsets=tuple(sorted(target_offsets, reverse=True)),
        exogenous_timestamp_offsets={
            name: tuple(sorted(offsets, reverse=True))
            for name, offsets in sorted(exogenous_offsets.items())
        },
        recipe_digest=frozen.digest,
        _terms=tuple(terms),
    )
    if max_history_hours is not None and plan.max_history_hours > max_history_hours:
        raise ValueError(
            f"feature_recipe needs {plan.max_history_hours}h of history for "
            f"{target}@{horizon_hours}h but the cohort only guarantees "
            f"{max_history_hours}h"
        )
    return plan


def recipe_training_summary(
    recipe: FrozenRecipe | Mapping[str, Any],
) -> dict[str, Any]:
    """The instance-free projection of what a recipe will read, per horizon.

    Folded into ``compiled_behavior_digest`` by the behavior compiler, so a
    frozen behavior records the reads it authorizes and any recipe edit moves
    the digest. Feature names carry ``STRUCTURAL_TARGET_PLACEHOLDER`` instead of
    a target name: read offsets and arity depend only on the horizon, and the
    concrete target set is already frozen by the ``dataset_task`` contract, so
    projecting it here would duplicate a fact without adding one.
    """

    frozen = recipe if isinstance(recipe, FrozenRecipe) else validate_feature_recipe(recipe)
    per_horizon: dict[str, Any] = {}
    for horizon in sorted(frozen.horizons):
        plan = compile_feature_plan(
            frozen,
            target=STRUCTURAL_TARGET_PLACEHOLDER,
            horizon_hours=horizon,
        )
        per_horizon[str(horizon)] = {
            "feature_names_template": list(plan.feature_names),
            "max_history_hours": plan.max_history_hours,
            "required_timestamp_offsets": list(plan.required_timestamp_offsets),
            "exogenous_timestamp_offsets": {
                column: list(offsets)
                for column, offsets in sorted(
                    plan.exogenous_timestamp_offsets.items()
                )
            },
            "residual_scale": frozen.residual_scale_for(
                STRUCTURAL_TARGET_PLACEHOLDER, horizon
            ),
        }
    return {
        "schema_version": FEATURE_RECIPE_SCHEMA_VERSION,
        "recipe_digest": frozen.digest,
        "term_count": frozen.term_count,
        "feature_arity": frozen.feature_arity,
        "model": {
            "kind": frozen.model["kind"],
            "alpha": frozen.ridge_alpha,
            "anchor": frozen.anchor,
        },
        "exogenous_columns": list(frozen.referenced_columns),
        "per_horizon": per_horizon,
    }


def recipe_grammar() -> dict[str, Any]:
    """The whitelist as data, for the program registry and Agent-facing catalog.

    One projection serves two consumers with opposite constraints. The program
    registry hashes it into ``catalog_digest``, so it has to be deterministic
    and complete. The sample Agent receives it inside
    ``plan.tools[*].parameters``, where ``sample_contracts._safe_value``
    rejects anything nested past ten levels — so ``allowed_ops`` is a mapping
    keyed by op name and per-parameter bounds live in a *flat* sibling map
    keyed ``"<op>.<parameter>"`` instead of nesting a list of objects inside
    each op. Both keyings come from ``RECIPE_PRIMITIVES``, so they cannot
    drift apart.
    """

    allowed_ops: dict[str, Any] = {}
    op_parameters: dict[str, Any] = {}
    for op in sorted(RECIPE_PRIMITIVES):
        primitive = RECIPE_PRIMITIVES[op]
        allowed_ops[op] = primitive.to_dict()
        for parameter in primitive.parameters:
            op_parameters[f"{op}.{parameter.name}"] = parameter.to_dict()
    return {
        "schema_version": FEATURE_RECIPE_SCHEMA_VERSION,
        "max_terms": MAX_RECIPE_TERMS,
        "max_lag_hours": MAX_RECIPE_LAG_HOURS,
        "max_rolling_window": MAX_ROLLING_WINDOW_HOURS,
        "allowed_ops": allowed_ops,
        "op_parameters": op_parameters,
        "model": {
            "kind": sorted(ALLOWED_MODEL_KINDS),
            "anchor": sorted(ALLOWED_MODEL_ANCHORS),
            "alpha": {"minimum": MODEL_ALPHA_MINIMUM, "maximum": MODEL_ALPHA_MAXIMUM},
        },
        "residual_scale": {
            "minimum": RESIDUAL_SCALE_MINIMUM,
            "maximum": RESIDUAL_SCALE_MAXIMUM,
        },
        "duplicate_terms": "rejected",
        "numeric_bounds_enforced_by": "host",
    }


__all__ = [
    "FEATURE_RECIPE_SCHEMA_VERSION",
    "FeaturePlan",
    "FrozenRecipe",
    "MAX_RECIPE_TERMS",
    "PrimitiveParameter",
    "PrimitiveSpec",
    "RECIPE_PRIMITIVES",
    "RECIPE_TOP_LEVEL_KEYS",
    "STRUCTURAL_TARGET_PLACEHOLDER",
    "compile_feature_plan",
    "recipe_digest",
    "recipe_grammar",
    "recipe_training_summary",
    "validate_feature_recipe",
]
