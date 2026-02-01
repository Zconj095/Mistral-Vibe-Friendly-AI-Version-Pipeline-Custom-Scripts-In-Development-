from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_PERMISSIONS = {"always", "ask", "never"}
VALID_RUN_TYPES = {"python", "shell"}


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    version: str | None
    parameters: dict[str, Any]
    run: dict[str, Any]
    permission: str
    source_path: Path


def load_action_specs(action_dir: Path) -> tuple[dict[str, ActionSpec], list[str]]:
    actions: dict[str, ActionSpec] = {}
    errors: list[str] = []

    if not action_dir.exists():
        return actions, errors

    for path in sorted(action_dir.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
            continue

        spec, error = _parse_action_spec(raw, path)
        if error:
            errors.append(error)
            continue
        if spec.name in actions:
            errors.append(f"{path.name}: duplicate action name '{spec.name}'")
            continue
        actions[spec.name] = spec

    return actions, errors


def apply_defaults(schema: dict[str, Any], payload: Any) -> Any:
    schema_type = schema.get("type")
    if schema_type == "object":
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            return payload
        properties = schema.get("properties", {})
        result = dict(payload)
        for key, subschema in properties.items():
            if key not in result and isinstance(subschema, dict):
                if "default" in subschema:
                    result[key] = subschema["default"]
        for key, value in list(result.items()):
            subschema = properties.get(key)
            if isinstance(subschema, dict):
                result[key] = apply_defaults(subschema, value)
        return result
    if schema_type == "array" and isinstance(payload, list):
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            return [apply_defaults(items_schema, item) for item in payload]
    return payload


def validate_args(schema: dict[str, Any], payload: Any) -> list[str]:
    return _validate_value(schema, payload, path="args")


def _parse_action_spec(raw: Any, path: Path) -> tuple[ActionSpec | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"{path.name}: action spec must be a JSON object"

    name = raw.get("name")
    description = raw.get("description")
    parameters = raw.get("parameters")
    run = raw.get("run")
    permission = raw.get("permission", "always")
    version = raw.get("version")

    if not isinstance(name, str) or not name.strip():
        return None, f"{path.name}: missing or invalid 'name'"
    if not isinstance(description, str) or not description.strip():
        return None, f"{path.name}: missing or invalid 'description'"
    if not isinstance(parameters, dict):
        return None, f"{path.name}: missing or invalid 'parameters'"
    if not isinstance(run, dict):
        return None, f"{path.name}: missing or invalid 'run'"
    if permission not in VALID_PERMISSIONS:
        return None, f"{path.name}: invalid permission '{permission}'"

    run_type = run.get("type")
    if run_type not in VALID_RUN_TYPES:
        return None, f"{path.name}: invalid run type '{run_type}'"
    if run_type == "python":
        if not isinstance(run.get("function"), str) or not run.get("function"):
            return None, f"{path.name}: python run requires 'function'"
        module_value = run.get("module")
        path_value = run.get("path")
        module_ok = isinstance(module_value, str) and module_value.strip()
        path_ok = isinstance(path_value, str) and path_value.strip()
        if not module_ok and not path_ok:
            return None, f"{path.name}: python run requires 'module' or 'path'"
    if run_type == "shell":
        argv = run.get("argv")
        command = run.get("command")
        if argv is None and command is None:
            return None, f"{path.name}: shell run requires 'argv' or 'command'"
        if argv is not None and not isinstance(argv, list):
            return None, f"{path.name}: shell 'argv' must be a list of strings"

    return (
        ActionSpec(
            name=name.strip(),
            description=description.strip(),
            version=version if isinstance(version, str) else None,
            parameters=parameters,
            run=run,
            permission=permission,
            source_path=path,
        ),
        None,
    )


def _validate_value(schema: dict[str, Any], value: Any, path: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(schema, dict):
        return errors

    schema_type = schema.get("type")
    if schema_type is None:
        return errors

    allowed_types = schema_type if isinstance(schema_type, list) else [schema_type]
    if "null" in allowed_types and value is None:
        return errors

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value must be one of {schema['enum']}")
        return errors

    if "object" in allowed_types and isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}.{key}: missing required field")
        properties = schema.get("properties", {})
        for key, val in value.items():
            subschema = properties.get(key)
            if subschema is None:
                if schema.get("additionalProperties") is False:
                    errors.append(f"{path}.{key}: unexpected field")
                continue
            errors.extend(_validate_value(subschema, val, f"{path}.{key}"))
        return errors

    if "array" in allowed_types and isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{path}: expected at least {min_items} items")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{path}: expected at most {max_items} items")
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for idx, item in enumerate(value):
                errors.extend(_validate_value(items_schema, item, f"{path}[{idx}]"))
        return errors

    if "string" in allowed_types and isinstance(value, str):
        min_len = schema.get("minLength")
        max_len = schema.get("maxLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append(f"{path}: expected length >= {min_len}")
        if isinstance(max_len, int) and len(value) > max_len:
            errors.append(f"{path}: expected length <= {max_len}")
        return errors

    if "integer" in allowed_types and isinstance(value, int) and not isinstance(
        value, bool
    ):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: expected >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: expected <= {maximum}")
        return errors

    if "number" in allowed_types and isinstance(value, (int, float)) and not isinstance(
        value, bool
    ):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, (int, float)) and value < minimum:
            errors.append(f"{path}: expected >= {minimum}")
        if isinstance(maximum, (int, float)) and value > maximum:
            errors.append(f"{path}: expected <= {maximum}")
        return errors

    if "boolean" in allowed_types and isinstance(value, bool):
        return errors

    errors.append(f"{path}: expected types {allowed_types}")
    return errors
