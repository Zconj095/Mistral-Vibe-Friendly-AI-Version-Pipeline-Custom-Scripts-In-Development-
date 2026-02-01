from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math
import statistics
from typing import TYPE_CHECKING, ClassVar, Any

from pydantic import BaseModel, Field

from vibe.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    ToolError,
    ToolPermission,
)
from vibe.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData

if TYPE_CHECKING:
    from vibe.core.types import ToolCallEvent, ToolResultEvent


DEFAULT_PROFILE_PATH = Path.home() / ".vibe" / "memory" / "meta_stats_profile.json"


@dataclass
class _ColumnBucket:
    numeric: list[float]
    categorical: dict[str, int]
    missing: int = 0


@dataclass
class _MeanTracker:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    def stdev(self) -> float:
        if self.count < 2:
            return 0.0
        return (self.m2 / (self.count - 1)) ** 0.5


class ContextMetaStatsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    max_items: int = Field(default=200, description="Maximum datasets to process.")
    max_source_bytes: int = Field(
        default=5_000_000, description="Maximum size per dataset (bytes)."
    )
    max_total_bytes: int = Field(
        default=20_000_000, description="Maximum total bytes across datasets."
    )
    max_rows: int = Field(default=50_000, description="Maximum rows per dataset.")
    default_format: str = Field(default="auto", description="auto, csv, tsv, json, jsonl.")
    default_sample_strategy: str = Field(
        default="head", description="head or uniform."
    )
    min_numeric_ratio: float = Field(
        default=0.6, description="Ratio of numeric values to treat column as numeric."
    )
    max_categories: int = Field(
        default=10, description="Maximum categorical values returned per column."
    )
    drift_effect_size: float = Field(
        default=0.5, description="Effect size threshold to flag drift."
    )
    drift_delta: float = Field(
        default=0.0, description="Absolute mean delta threshold when stdev is zero."
    )
    anomaly_z: float = Field(
        default=2.5, description="Z-score threshold for anomalies across datasets."
    )
    profile_path: Path = Field(
        default=DEFAULT_PROFILE_PATH,
        description="Path to the persistent meta-profile JSON.",
    )


class ContextMetaStatsState(BaseToolState):
    pass


class DatasetInput(BaseModel):
    id: str | None = Field(default=None, description="Optional dataset id.")
    name: str | None = Field(default=None, description="Optional dataset name.")
    content: str | None = Field(default=None, description="Inline dataset content.")
    path: str | None = Field(default=None, description="Path to a dataset file.")
    format: str | None = Field(default=None, description="auto, csv, tsv, json, jsonl.")
    delimiter: str | None = Field(default=None, description="Override csv delimiter.")
    include_columns: list[str] | None = Field(
        default=None, description="Columns to include for this dataset."
    )
    exclude_columns: list[str] | None = Field(
        default=None, description="Columns to exclude for this dataset."
    )
    numeric_columns: list[str] | None = Field(
        default=None, description="Force columns to be treated as numeric."
    )


class ContextMetaStatsArgs(BaseModel):
    action: str | None = Field(
        default="analyze", description="analyze, update, profile, or clear."
    )
    datasets: list[DatasetInput] | None = Field(
        default=None, description="Datasets to analyze."
    )
    format: str | None = Field(default=None, description="Override default format.")
    sample_strategy: str | None = Field(
        default=None, description="Override sample strategy: head or uniform."
    )
    max_rows: int | None = Field(default=None, description="Override max rows.")
    min_numeric_ratio: float | None = Field(
        default=None, description="Override numeric ratio threshold."
    )
    max_categories: int | None = Field(
        default=None, description="Override max categorical values returned."
    )
    include_columns: list[str] | None = Field(
        default=None, description="Columns to include globally."
    )
    exclude_columns: list[str] | None = Field(
        default=None, description="Columns to exclude globally."
    )
    numeric_columns: list[str] | None = Field(
        default=None, description="Force columns to be treated as numeric."
    )
    drift_effect_size: float | None = Field(
        default=None, description="Override drift effect size threshold."
    )
    drift_delta: float | None = Field(
        default=None, description="Override drift absolute delta threshold."
    )
    anomaly_z: float | None = Field(
        default=None, description="Override anomaly z-score threshold."
    )
    profile_path: str | None = Field(
        default=None, description="Override profile path."
    )


class NumericColumnStats(BaseModel):
    column: str
    count: int
    missing: int
    mean: float | None
    median: float | None
    stdev: float | None
    minimum: float | None
    maximum: float | None
    p25: float | None
    p75: float | None
    ci95_low: float | None
    ci95_high: float | None


class CategoryValue(BaseModel):
    value: str
    count: int
    ratio: float


class CategoricalColumnStats(BaseModel):
    column: str
    count: int
    missing: int
    unique: int
    top_values: list[CategoryValue]


class DatasetSummary(BaseModel):
    index: int
    id: str | None
    name: str | None
    source: str | None
    rows: int
    rows_total: int
    truncated: bool
    columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_cells: int
    numeric_stats: list[NumericColumnStats]
    categorical_stats: list[CategoricalColumnStats]
    warnings: list[str]


class DriftSignal(BaseModel):
    column: str
    dataset_index_a: int
    dataset_index_b: int
    delta_mean: float
    effect_size: float
    relative_change: float | None
    drift_flag: bool


class AnomalySignal(BaseModel):
    column: str
    dataset_index: int
    mean: float
    z_score: float
    flagged: bool


class ModelFit(BaseModel):
    model: str
    r2: float
    slope: float | None
    intercept: float | None


class ModelSelection(BaseModel):
    column: str
    best_model: str
    fits: list[ModelFit]


class MetaProfileColumn(BaseModel):
    column: str
    dataset_count: int
    total_count: int
    mean_of_means: float
    stdev_of_means: float
    total_sum: float | None
    total_min: float | None
    total_max: float | None
    overall_mean: float | None


class ContextMetaStatsResult(BaseModel):
    datasets: list[DatasetSummary]
    drift: list[DriftSignal]
    anomalies: list[AnomalySignal]
    model_selection: list[ModelSelection]
    meta_profile: list[MetaProfileColumn]
    dataset_count: int
    drift_count: int
    anomaly_count: int
    model_count: int
    updated: bool
    truncated: bool
    errors: list[str]

class ContextMetaStats(
    BaseTool[
        ContextMetaStatsArgs,
        ContextMetaStatsResult,
        ContextMetaStatsConfig,
        ContextMetaStatsState,
    ],
    ToolUIData[ContextMetaStatsArgs, ContextMetaStatsResult],
):
    description: ClassVar[str] = (
        "Meta-statistical learning across datasets with drift and model signals."
    )

    async def run(self, args: ContextMetaStatsArgs) -> ContextMetaStatsResult:
        action = (args.action or "analyze").strip().lower()
        if action not in {"analyze", "update", "profile", "clear"}:
            raise ToolError("action must be analyze, update, profile, or clear.")

        profile_path = self._resolve_profile_path(args.profile_path)
        if action == "profile":
            profile = self._load_profile(profile_path)
            return ContextMetaStatsResult(
                datasets=[],
                drift=[],
                anomalies=[],
                model_selection=[],
                meta_profile=profile,
                dataset_count=0,
                drift_count=0,
                anomaly_count=0,
                model_count=0,
                updated=False,
                truncated=False,
                errors=[],
            )

        if action == "clear":
            removed = self._clear_profile(profile_path)
            return ContextMetaStatsResult(
                datasets=[],
                drift=[],
                anomalies=[],
                model_selection=[],
                meta_profile=[],
                dataset_count=0,
                drift_count=0,
                anomaly_count=0,
                model_count=0,
                updated=removed,
                truncated=False,
                errors=[],
            )

        datasets = args.datasets or []
        if not datasets:
            raise ToolError("datasets is required for analyze/update.")

        max_items = self.config.max_items
        if len(datasets) > max_items:
            raise ToolError(f"datasets exceeds max_items ({len(datasets)} > {max_items}).")

        max_source_bytes = self.config.max_source_bytes
        max_total_bytes = self.config.max_total_bytes
        max_rows = args.max_rows if args.max_rows is not None else self.config.max_rows
        if max_rows <= 0:
            raise ToolError("max_rows must be a positive integer.")

        default_format = args.format or self.config.default_format
        sample_strategy = (args.sample_strategy or self.config.default_sample_strategy).strip().lower()
        if sample_strategy not in {"head", "uniform"}:
            raise ToolError("sample_strategy must be head or uniform.")

        min_numeric_ratio = (
            args.min_numeric_ratio
            if args.min_numeric_ratio is not None
            else self.config.min_numeric_ratio
        )
        max_categories = (
            args.max_categories
            if args.max_categories is not None
            else self.config.max_categories
        )
        if min_numeric_ratio < 0 or min_numeric_ratio > 1:
            raise ToolError("min_numeric_ratio must be between 0 and 1.")
        if max_categories < 0:
            raise ToolError("max_categories must be >= 0.")

        drift_effect_size = (
            args.drift_effect_size
            if args.drift_effect_size is not None
            else self.config.drift_effect_size
        )
        drift_delta = (
            args.drift_delta
            if args.drift_delta is not None
            else self.config.drift_delta
        )
        anomaly_z = (
            args.anomaly_z
            if args.anomaly_z is not None
            else self.config.anomaly_z
        )
        if drift_effect_size < 0:
            raise ToolError("drift_effect_size must be >= 0.")
        if drift_delta < 0:
            raise ToolError("drift_delta must be >= 0.")
        if anomaly_z < 0:
            raise ToolError("anomaly_z must be >= 0.")

        global_include = self._normalize_columns(args.include_columns)
        global_exclude = self._normalize_columns(args.exclude_columns)
        global_numeric = self._normalize_columns(args.numeric_columns)

        datasets_out: list[DatasetSummary] = []
        errors: list[str] = []
        total_bytes = 0
        truncated = False

        for idx, dataset in enumerate(datasets, start=1):
            try:
                content, source, size_bytes = self._load_dataset_content(
                    dataset, max_source_bytes
                )
                if content is None:
                    raise ToolError("Dataset has no content.")
                if size_bytes is not None:
                    if total_bytes + size_bytes > max_total_bytes:
                        truncated = True
                        break
                    total_bytes += size_bytes

                resolved_format = self._resolve_format(dataset, content, default_format)
                rows, total_rows, row_truncated = self._parse_rows(
                    content,
                    resolved_format,
                    dataset.delimiter,
                    max_rows,
                    sample_strategy,
                )
                if row_truncated:
                    truncated = True

                include_cols = self._merge_columns(global_include, dataset.include_columns)
                exclude_cols = self._merge_columns(global_exclude, dataset.exclude_columns)
                numeric_cols = self._merge_columns(global_numeric, dataset.numeric_columns)

                summary = self._summarize_dataset(
                    dataset,
                    rows,
                    total_rows,
                    row_truncated,
                    include_cols,
                    exclude_cols,
                    numeric_cols,
                    min_numeric_ratio,
                    max_categories,
                )
                summary.index = len(datasets_out) + 1
                datasets_out.append(summary)
            except ToolError as exc:
                errors.append(f"dataset[{idx}]: {exc}")
            except Exception as exc:
                errors.append(f"dataset[{idx}]: {exc}")

        if not datasets_out:
            raise ToolError("No valid datasets to process.")

        drift = self._build_drift_signals(
            datasets_out,
            drift_effect_size,
            drift_delta,
        )
        anomalies = self._build_anomaly_signals(datasets_out, anomaly_z)
        model_selection = self._build_model_selection(datasets_out)

        meta_profile: list[MetaProfileColumn] = []
        updated = False
        if action == "update":
            meta_profile = self._update_profile(profile_path, datasets_out)
            updated = True

        return ContextMetaStatsResult(
            datasets=datasets_out,
            drift=drift,
            anomalies=anomalies,
            model_selection=model_selection,
            meta_profile=meta_profile,
            dataset_count=len(datasets_out),
            drift_count=len(drift),
            anomaly_count=len(anomalies),
            model_count=len(model_selection),
            updated=updated,
            truncated=truncated,
            errors=errors,
        )

    def _resolve_profile_path(self, raw: str | None) -> Path:
        path = Path(raw).expanduser() if raw else self.config.profile_path
        if not path.is_absolute():
            path = self.config.effective_workdir / path
        return path.resolve()

    def _clear_profile(self, path: Path) -> bool:
        if path.exists():
            path.unlink()
            return True
        return False

    def _load_profile(self, path: Path) -> list[MetaProfileColumn]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception as exc:
            raise ToolError(f"Failed to load profile: {exc}") from exc

        columns = data.get("columns") if isinstance(data, dict) else None
        if not isinstance(columns, dict):
            return []
        return self._profile_columns(columns)

    def _update_profile(
        self, path: Path, datasets: list[DatasetSummary]
    ) -> list[MetaProfileColumn]:
        store = {"version": 1, "columns": {}}
        if path.exists():
            try:
                store = json.loads(path.read_text("utf-8"))
            except Exception:
                store = {"version": 1, "columns": {}}
        if not isinstance(store, dict):
            store = {"version": 1, "columns": {}}
        columns = store.get("columns")
        if not isinstance(columns, dict):
            columns = {}
            store["columns"] = columns

        for dataset in datasets:
            for stat in dataset.numeric_stats:
                if stat.count <= 0 or stat.mean is None:
                    continue
                entry = columns.get(stat.column)
                if not isinstance(entry, dict):
                    entry = {}
                tracker = _MeanTracker(
                    count=int(entry.get("dataset_count", 0)),
                    mean=float(entry.get("mean_of_means", 0.0)),
                    m2=float(entry.get("m2", 0.0)),
                )
                tracker.update(float(stat.mean))

                total_count = int(entry.get("total_count", 0)) + int(stat.count)
                total_sum = float(entry.get("total_sum", 0.0)) + float(stat.mean) * int(stat.count)
                total_min = entry.get("total_min")
                total_max = entry.get("total_max")
                if stat.minimum is not None:
                    total_min = stat.minimum if total_min is None else min(float(total_min), stat.minimum)
                if stat.maximum is not None:
                    total_max = stat.maximum if total_max is None else max(float(total_max), stat.maximum)

                columns[stat.column] = {
                    "dataset_count": tracker.count,
                    "mean_of_means": tracker.mean,
                    "m2": tracker.m2,
                    "total_count": total_count,
                    "total_sum": total_sum,
                    "total_min": total_min,
                    "total_max": total_max,
                }

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(store, indent=2), "utf-8")
        return self._profile_columns(columns)

    def _profile_columns(self, columns: dict[str, Any]) -> list[MetaProfileColumn]:
        output: list[MetaProfileColumn] = []
        for name, entry in columns.items():
            if not isinstance(entry, dict):
                continue
            dataset_count = int(entry.get("dataset_count", 0))
            mean_of_means = float(entry.get("mean_of_means", 0.0))
            m2 = float(entry.get("m2", 0.0))
            total_count = int(entry.get("total_count", 0))
            total_sum = entry.get("total_sum")
            total_min = entry.get("total_min")
            total_max = entry.get("total_max")
            stdev_of_means = 0.0
            if dataset_count > 1:
                stdev_of_means = (m2 / (dataset_count - 1)) ** 0.5
            overall_mean = None
            if total_sum is not None and total_count > 0:
                overall_mean = float(total_sum) / total_count

            output.append(
                MetaProfileColumn(
                    column=name,
                    dataset_count=dataset_count,
                    total_count=total_count,
                    mean_of_means=round(mean_of_means, 6),
                    stdev_of_means=round(stdev_of_means, 6),
                    total_sum=float(total_sum) if total_sum is not None else None,
                    total_min=float(total_min) if total_min is not None else None,
                    total_max=float(total_max) if total_max is not None else None,
                    overall_mean=round(overall_mean, 6) if overall_mean is not None else None,
                )
            )

        output.sort(key=lambda item: item.column)
        return output

    def _load_dataset_content(
        self, dataset: DatasetInput, max_source_bytes: int
    ) -> tuple[str | None, str | None, int | None]:
        if dataset.content and dataset.path:
            raise ToolError("Provide content or path, not both.")

        if dataset.path:
            path = Path(dataset.path).expanduser()
            if not path.is_absolute():
                path = self.config.effective_workdir / path
            path = path.resolve()
            if not path.exists():
                raise ToolError(f"Path not found: {path}")
            if path.is_dir():
                raise ToolError(f"Path is a directory, not a file: {path}")
            size = path.stat().st_size
            if size > max_source_bytes:
                raise ToolError(
                    f"{path} exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return path.read_text("utf-8", errors="ignore"), str(path), size

        if dataset.content is not None:
            size = len(dataset.content.encode("utf-8"))
            if size > max_source_bytes:
                raise ToolError(
                    f"content exceeds max_source_bytes ({size} > {max_source_bytes})."
                )
            return dataset.content, None, size

        return None, None, None

    def _resolve_format(self, dataset: DatasetInput, content: str, default_format: str) -> str:
        fmt = (dataset.format or default_format or "auto").strip().lower()
        if fmt != "auto":
            return fmt

        if dataset.path:
            ext = Path(dataset.path).suffix.lower()
            if ext == ".tsv":
                return "tsv"
            if ext == ".jsonl":
                return "jsonl"
            if ext == ".json":
                return "json"
            if ext == ".csv":
                return "csv"

        stripped = content.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(content)
                return "json"
            except Exception:
                return "jsonl"

        first_line = content.splitlines()[0] if content else ""
        if "\t" in first_line:
            return "tsv"
        return "csv"

    def _parse_rows(
        self,
        content: str,
        fmt: str,
        delimiter: str | None,
        max_rows: int,
        sample_strategy: str,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        rows: list[dict[str, Any]] = []
        if fmt in {"csv", "tsv"}:
            delim = delimiter if delimiter else ("\t" if fmt == "tsv" else ",")
            reader = csv.DictReader(content.splitlines(), delimiter=delim)
            for row in reader:
                rows.append(dict(row))
        elif fmt == "jsonl":
            for line in content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                value = json.loads(stripped)
                rows.append(self._row_from_value(value))
        elif fmt == "json":
            value = json.loads(content)
            rows = self._rows_from_json(value)
        else:
            raise ToolError("format must be auto, csv, tsv, json, or jsonl.")

        total_rows = len(rows)
        truncated = False
        if max_rows > 0 and total_rows > max_rows:
            rows = self._sample_rows(rows, max_rows, sample_strategy)
            truncated = True

        return rows, total_rows, truncated

    def _row_from_value(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        return {"value": value}

    def _rows_from_json(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, list):
            return [self._row_from_value(item) for item in value]
        if isinstance(value, dict):
            if isinstance(value.get("rows"), list):
                return [self._row_from_value(item) for item in value["rows"]]
            if isinstance(value.get("data"), list):
                return [self._row_from_value(item) for item in value["data"]]
            if all(isinstance(item, list) for item in value.values()):
                keys = list(value.keys())
                max_len = max((len(value[key]) for key in keys), default=0)
                rows: list[dict[str, Any]] = []
                for idx in range(max_len):
                    row = {key: value[key][idx] if idx < len(value[key]) else None for key in keys}
                    rows.append(row)
                return rows
            return [value]
        return [{"value": value}]

    def _sample_rows(
        self, rows: list[dict[str, Any]], max_rows: int, strategy: str
    ) -> list[dict[str, Any]]:
        if max_rows <= 0:
            return rows
        if strategy == "head":
            return rows[:max_rows]
        step = max(1, len(rows) // max_rows)
        sampled = rows[::step]
        return sampled[:max_rows]

    def _summarize_dataset(
        self,
        dataset: DatasetInput,
        rows: list[dict[str, Any]],
        total_rows: int,
        truncated: bool,
        include_columns: list[str],
        exclude_columns: list[str],
        numeric_columns: list[str],
        min_numeric_ratio: float,
        max_categories: int,
    ) -> DatasetSummary:
        columns = self._collect_columns(rows)
        columns = self._filter_columns(columns, include_columns, exclude_columns)
        if not columns:
            return DatasetSummary(
                index=0,
                id=dataset.id,
                name=dataset.name,
                source=dataset.path,
                rows=len(rows),
                rows_total=total_rows,
                truncated=truncated,
                columns=[],
                numeric_columns=[],
                categorical_columns=[],
                missing_cells=0,
                numeric_stats=[],
                categorical_stats=[],
                warnings=["No columns detected."],
            )

        buckets: dict[str, _ColumnBucket] = {
            column: _ColumnBucket(numeric=[], categorical={}, missing=0) for column in columns
        }
        missing_cells = 0

        for row in rows:
            for column in columns:
                value = row.get(column)
                bucket = buckets[column]
                if self._ingest_value(bucket, value):
                    missing_cells += 1

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        numeric_stats: list[NumericColumnStats] = []
        categorical_stats: list[CategoricalColumnStats] = []

        for column in columns:
            bucket = buckets[column]
            numeric_count = len(bucket.numeric)
            categorical_count = sum(bucket.categorical.values())
            forced_numeric = column in numeric_columns

            if forced_numeric:
                is_numeric = True
            elif numeric_count == 0 and categorical_count == 0:
                is_numeric = False
            else:
                ratio = numeric_count / max(numeric_count + categorical_count, 1)
                is_numeric = numeric_count > 0 and ratio >= min_numeric_ratio

            if is_numeric:
                numeric_cols.append(column)
                numeric_stats.append(self._numeric_stats(column, bucket))
            else:
                categorical_cols.append(column)
                categorical_stats.append(self._categorical_stats(column, bucket, max_categories))

        warnings: list[str] = []
        if truncated:
            warnings.append("Rows truncated to max_rows.")
        if not numeric_cols:
            warnings.append("No numeric columns detected.")
        if not categorical_cols:
            warnings.append("No categorical columns detected.")

        return DatasetSummary(
            index=0,
            id=dataset.id,
            name=dataset.name,
            source=dataset.path,
            rows=len(rows),
            rows_total=total_rows,
            truncated=truncated,
            columns=columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            missing_cells=missing_cells,
            numeric_stats=numeric_stats,
            categorical_stats=categorical_stats,
            warnings=warnings,
        )

    def _collect_columns(self, rows: list[dict[str, Any]]) -> list[str]:
        columns: set[str] = set()
        for row in rows:
            columns.update(str(key) for key in row.keys())
        return sorted(columns)

    def _normalize_columns(self, columns: list[str] | None) -> list[str]:
        if not columns:
            return []
        return [col.strip() for col in columns if col and col.strip()]

    def _merge_columns(self, global_cols: list[str], local_cols: list[str] | None) -> list[str]:
        merged = list(global_cols)
        if local_cols:
            merged.extend(self._normalize_columns(local_cols))
        return sorted(set(merged))

    def _filter_columns(
        self, columns: list[str], include_cols: list[str], exclude_cols: list[str]
    ) -> list[str]:
        filtered = columns
        if include_cols:
            include_set = set(include_cols)
            filtered = [col for col in filtered if col in include_set]
        if exclude_cols:
            exclude_set = set(exclude_cols)
            filtered = [col for col in filtered if col not in exclude_set]
        return filtered

    def _ingest_value(self, bucket: _ColumnBucket, value: Any) -> bool:
        if value is None:
            bucket.missing += 1
            return True
        if isinstance(value, bool):
            key = "true" if value else "false"
            bucket.categorical[key] = bucket.categorical.get(key, 0) + 1
            return False
        if isinstance(value, (int, float)):
            bucket.numeric.append(float(value))
            return False
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                bucket.missing += 1
                return True
            numeric = self._try_float(cleaned)
            if numeric is not None:
                bucket.numeric.append(numeric)
                return False
            bucket.categorical[cleaned] = bucket.categorical.get(cleaned, 0) + 1
            return False
        text = json.dumps(value, ensure_ascii=True)
        bucket.categorical[text] = bucket.categorical.get(text, 0) + 1
        return False

    def _try_float(self, value: str) -> float | None:
        cleaned = value.replace(",", "")
        try:
            number = float(cleaned)
        except ValueError:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number

    def _numeric_stats(self, column: str, bucket: _ColumnBucket) -> NumericColumnStats:
        values = bucket.numeric
        count = len(values)
        missing = bucket.missing
        if count == 0:
            return NumericColumnStats(
                column=column,
                count=0,
                missing=missing,
                mean=None,
                median=None,
                stdev=None,
                minimum=None,
                maximum=None,
                p25=None,
                p75=None,
                ci95_low=None,
                ci95_high=None,
            )

        sorted_values = sorted(values)
        mean = statistics.mean(sorted_values)
        median = statistics.median(sorted_values)
        stdev = statistics.stdev(sorted_values) if count > 1 else 0.0
        minimum = sorted_values[0]
        maximum = sorted_values[-1]
        p25 = self._percentile(sorted_values, 0.25)
        p75 = self._percentile(sorted_values, 0.75)

        ci95_low = None
        ci95_high = None
        if count > 1:
            stderr = stdev / math.sqrt(count)
            ci95_low = mean - 1.96 * stderr
            ci95_high = mean + 1.96 * stderr
        else:
            ci95_low = mean
            ci95_high = mean

        return NumericColumnStats(
            column=column,
            count=count,
            missing=missing,
            mean=round(mean, 6),
            median=round(median, 6),
            stdev=round(stdev, 6),
            minimum=round(minimum, 6),
            maximum=round(maximum, 6),
            p25=round(p25, 6) if p25 is not None else None,
            p75=round(p75, 6) if p75 is not None else None,
            ci95_low=round(ci95_low, 6) if ci95_low is not None else None,
            ci95_high=round(ci95_high, 6) if ci95_high is not None else None,
        )

    def _categorical_stats(
        self, column: str, bucket: _ColumnBucket, max_categories: int
    ) -> CategoricalColumnStats:
        counts = dict(bucket.categorical)
        for numeric in bucket.numeric:
            key = str(numeric)
            counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values())
        missing = bucket.missing
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if max_categories > 0:
            ordered = ordered[:max_categories]
        top_values = [
            CategoryValue(
                value=value,
                count=count,
                ratio=round(count / total, 6) if total > 0 else 0.0,
            )
            for value, count in ordered
        ]

        return CategoricalColumnStats(
            column=column,
            count=total,
            missing=missing,
            unique=len(counts),
            top_values=top_values,
        )

    def _percentile(self, values: list[float], pct: float) -> float | None:
        if not values:
            return None
        if pct <= 0:
            return values[0]
        if pct >= 1:
            return values[-1]
        if len(values) == 1:
            return values[0]
        index = (len(values) - 1) * pct
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return values[lower]
        weight = index - lower
        return values[lower] + (values[upper] - values[lower]) * weight

    def _build_drift_signals(
        self,
        datasets: list[DatasetSummary],
        drift_effect_size: float,
        drift_delta: float,
    ) -> list[DriftSignal]:
        signals: list[DriftSignal] = []
        for idx in range(1, len(datasets)):
            prev = datasets[idx - 1]
            curr = datasets[idx]
            prev_stats = {stat.column: stat for stat in prev.numeric_stats}
            curr_stats = {stat.column: stat for stat in curr.numeric_stats}
            common = sorted(set(prev_stats.keys()) & set(curr_stats.keys()))
            for column in common:
                a = prev_stats[column]
                b = curr_stats[column]
                if a.mean is None or b.mean is None:
                    continue
                delta = float(b.mean) - float(a.mean)
                effect_size = self._effect_size(a, b, delta)
                relative = None
                if a.mean != 0:
                    relative = delta / float(a.mean)
                drift_flag = abs(effect_size) >= drift_effect_size
                if drift_delta > 0 and abs(effect_size) == 0 and abs(delta) >= drift_delta:
                    drift_flag = True

                signals.append(
                    DriftSignal(
                        column=column,
                        dataset_index_a=prev.index,
                        dataset_index_b=curr.index,
                        delta_mean=round(delta, 6),
                        effect_size=round(effect_size, 6),
                        relative_change=round(relative, 6) if relative is not None else None,
                        drift_flag=drift_flag,
                    )
                )

        return signals

    def _effect_size(
        self, left: NumericColumnStats, right: NumericColumnStats, delta: float
    ) -> float:
        if left.stdev is None or right.stdev is None:
            return 0.0
        n1 = max(left.count, 1)
        n2 = max(right.count, 1)
        if n1 + n2 - 2 <= 0:
            return 0.0
        pooled = ((n1 - 1) * left.stdev ** 2 + (n2 - 1) * right.stdev ** 2) / (n1 + n2 - 2)
        if pooled <= 0:
            return 0.0
        return delta / math.sqrt(pooled)

    def _build_anomaly_signals(
        self, datasets: list[DatasetSummary], anomaly_z: float
    ) -> list[AnomalySignal]:
        column_means: dict[str, list[tuple[int, float]]] = {}
        for dataset in datasets:
            for stat in dataset.numeric_stats:
                if stat.mean is None:
                    continue
                column_means.setdefault(stat.column, []).append((dataset.index, float(stat.mean)))

        signals: list[AnomalySignal] = []
        for column, values in column_means.items():
            if len(values) < 2:
                continue
            means = [value for _, value in values]
            overall_mean = statistics.mean(means)
            stdev = statistics.stdev(means) if len(means) > 1 else 0.0
            if stdev == 0:
                continue
            for dataset_index, mean in values:
                z = (mean - overall_mean) / stdev
                flagged = abs(z) >= anomaly_z
                signals.append(
                    AnomalySignal(
                        column=column,
                        dataset_index=dataset_index,
                        mean=round(mean, 6),
                        z_score=round(z, 6),
                        flagged=flagged,
                    )
                )

        signals.sort(key=lambda item: (-abs(item.z_score), item.column, item.dataset_index))
        return signals

    def _build_model_selection(
        self, datasets: list[DatasetSummary]
    ) -> list[ModelSelection]:
        column_means: dict[str, list[float]] = {}
        for dataset in datasets:
            stats = {stat.column: stat for stat in dataset.numeric_stats}
            for column, stat in stats.items():
                if stat.mean is None:
                    continue
                column_means.setdefault(column, []).append(float(stat.mean))

        selections: list[ModelSelection] = []
        for column, means in column_means.items():
            if len(means) < 2:
                continue
            fits = self._fit_models(means)
            if not fits:
                continue
            best = max(fits, key=lambda fit: fit.r2)
            selections.append(
                ModelSelection(
                    column=column,
                    best_model=best.model,
                    fits=fits,
                )
            )

        selections.sort(key=lambda item: item.column)
        return selections

    def _fit_models(self, means: list[float]) -> list[ModelFit]:
        x_values = [float(idx + 1) for idx in range(len(means))]
        y_values = means
        y_mean = statistics.mean(y_values)
        sst = sum((value - y_mean) ** 2 for value in y_values)
        fits: list[ModelFit] = []

        constant_r2 = 1.0 if sst == 0 else 0.0
        fits.append(
            ModelFit(
                model="constant",
                r2=round(constant_r2, 6),
                slope=None,
                intercept=round(y_mean, 6),
            )
        )

        slope, intercept = self._linear_fit(x_values, y_values)
        preds = [intercept + slope * x for x in x_values]
        sse = sum((y - pred) ** 2 for y, pred in zip(y_values, preds))
        r2 = 1.0 if sst == 0 else 1 - (sse / sst)
        fits.append(
            ModelFit(
                model="linear",
                r2=round(r2, 6),
                slope=round(slope, 6),
                intercept=round(intercept, 6),
            )
        )

        return fits

    def _linear_fit(self, x_values: list[float], y_values: list[float]) -> tuple[float, float]:
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        ss_xx = sum((x - x_mean) ** 2 for x in x_values)
        if ss_xx == 0:
            return 0.0, y_mean
        ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        slope = ss_xy / ss_xx
        intercept = y_mean - slope * x_mean
        return slope, intercept

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        if not isinstance(event.args, ContextMetaStatsArgs):
            return ToolCallDisplay(summary="context_meta_stats")

        summary = "context_meta_stats"
        return ToolCallDisplay(
            summary=summary,
            details={
                "action": event.args.action,
                "dataset_count": len(event.args.datasets or []),
                "format": event.args.format,
                "max_rows": event.args.max_rows,
                "sample_strategy": event.args.sample_strategy,
                "drift_effect_size": event.args.drift_effect_size,
                "anomaly_z": event.args.anomaly_z,
            },
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if not isinstance(event.result, ContextMetaStatsResult):
            return ToolResultDisplay(
                success=False, message=event.error or event.skip_reason or "No result"
            )

        message = (
            f"Processed {event.result.dataset_count} dataset(s) with "
            f"{event.result.drift_count} drift signal(s)"
        )
        warnings = event.result.errors[:]
        if event.result.truncated:
            warnings.append("Output truncated by size or row limits")

        return ToolResultDisplay(
            success=not bool(event.result.errors),
            message=message,
            warnings=warnings,
            details={
                "dataset_count": event.result.dataset_count,
                "drift_count": event.result.drift_count,
                "anomaly_count": event.result.anomaly_count,
                "model_count": event.result.model_count,
                "updated": event.result.updated,
                "truncated": event.result.truncated,
                "datasets": event.result.datasets,
                "drift": event.result.drift,
                "anomalies": event.result.anomalies,
                "model_selection": event.result.model_selection,
                "meta_profile": event.result.meta_profile,
                "errors": event.result.errors,
            },
        )

    @classmethod
    def get_status_text(cls) -> str:
        return "Analyzing meta-statistics"
