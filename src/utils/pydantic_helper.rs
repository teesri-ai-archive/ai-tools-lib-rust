use serde::Serialize;
use serde_json::{Map, Value};

/// Marker trait to describe fields that should be excluded when `SkipJsonSchema` is enforced.
pub trait SkipJsonSchema {
    /// Returns the nested map describing which fields to exclude.
    fn skip_json_schema_fields(&self) -> Option<Value>;
}

/// Build an exclusion dictionary mirroring `ai_tools.utils.pydantic_helper.build_exclude_dict`.
pub fn build_exclude_dict<T>(instance: &T) -> Option<Value>
where
    T: SkipJsonSchema + ?Sized,
{
    instance.skip_json_schema_fields()
}

/// Returns true if a field is marked as skipped inside an exclusion map.
pub fn is_field_skipped(exclude_map: Option<&Value>, field_name: &str) -> bool {
    match exclude_map {
        Some(Value::Object(map)) => map.get(field_name) == Some(&Value::Bool(true)),
        _ => false,
    }
}

/// Serialize the instance, optionally removing `SkipJsonSchema` fields when `enforce_schema` is `true`.
pub fn gated_serializer<T>(instance: &T, enforce_schema: bool) -> serde_json::Result<Value>
where
    T: Serialize + SkipJsonSchema + ?Sized,
{
    let mut serialized = serde_json::to_value(instance)?;
    if enforce_schema && let Some(exclude_map) = build_exclude_dict(instance) {
        apply_exclude(&mut serialized, &exclude_map);
    }
    Ok(serialized)
}

fn apply_exclude(target: &mut Value, exclude: &Value) {
    match (target, exclude) {
        (Value::Object(obj), Value::Object(rules)) => {
            for (key, rule) in rules {
                match (obj.get_mut(key), rule) {
                    (Some(_), Value::Bool(true)) => {
                        obj.remove(key);
                    }
                    (Some(value), recursive @ Value::Object(_)) => {
                        apply_exclude(value, recursive);
                        if is_empty(value) {
                            obj.remove(key);
                        }
                    }
                    _ => {}
                }
            }
        }
        (Value::Array(arr), Value::Object(rules)) => {
            for (idx_str, rule) in rules {
                if let Ok(idx) = idx_str.parse::<usize>()
                    && let Some(value) = arr.get_mut(idx)
                {
                    match rule {
                        Value::Bool(true) => {
                            *value = Value::Null;
                        }
                        recursive @ Value::Object(_) => {
                            apply_exclude(value, recursive);
                        }
                        _ => {}
                    }
                }
            }
        }
        _ => {}
    }
}

fn is_empty(value: &Value) -> bool {
    matches!(value, Value::Object(map) if map.is_empty())
}

impl<T: SkipJsonSchema> SkipJsonSchema for Vec<T> {
    fn skip_json_schema_fields(&self) -> Option<Value> {
        let mut map = Map::new();
        for (idx, item) in self.iter().enumerate() {
            if let Some(entry) = item.skip_json_schema_fields() {
                map.insert(idx.to_string(), entry);
            }
        }
        if map.is_empty() {
            None
        } else {
            Some(Value::Object(map))
        }
    }
}

impl<T: SkipJsonSchema> SkipJsonSchema for Option<T> {
    fn skip_json_schema_fields(&self) -> Option<Value> {
        self.as_ref()
            .and_then(|value| value.skip_json_schema_fields())
    }
}
