use include_dir::{Dir, include_dir};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
};
use tera::{Context, Tera};
use thiserror::Error;

use crate::prompt_manager::templates::PromptTemplate;

static PROMPTS_DIR: Dir = include_dir!("../flixie-prompts/prompts");
static PROMPT_CACHE: Lazy<Mutex<HashMap<PromptTemplate, PromptTemplateData>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

static TEMPLATE_VAR_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)").unwrap());
static FOR_VAR_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\{\%\s*for\s+[a-zA-Z_][a-zA-Z0-9_]*\s+in\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap()
});
static BLOCK_VAR_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\{\%\s*(?:if|elif|set)\s+([a-zA-Z_][a-zA-Z0-9_]*)").unwrap());

#[derive(Debug, Error)]
pub enum PromptTemplateError {
    #[error("prompt '{0}' not found")]
    NotFound(String),
    #[error(
        "rules for prompt '{prompt}' do not match template variables (missing {missing:?}, extra {extra:?})"
    )]
    VariableMismatch {
        prompt: String,
        missing: Vec<String>,
        extra: Vec<String>,
    },
    #[error("failed to parse variables for prompt {prompt}: {detail}")]
    VariablesParse { prompt: String, detail: String },
    #[error("template rendering failed for prompt {prompt}: {detail}")]
    Rendering { prompt: String, detail: String },
    #[error("variable '{name}' must match type '{expected_type}' per variables.yml ({detail})")]
    VariableValueType {
        name: String,
        expected_type: String,
        detail: String,
    },
}

/// Build prompt variables from string values (each becomes a JSON string). Matches Python callers
/// that only pass text while `variables.yml` declares `type: str`.
pub fn prompt_variables_from_str_map(map: &HashMap<String, String>) -> HashMap<String, Value> {
    map.iter()
        .map(|(k, v)| (k.clone(), Value::String(v.clone())))
        .collect()
}

fn json_is_integer(v: &Value) -> bool {
    v.as_i64().is_some() || v.as_u64().is_some()
}

fn validate_json_value_type(v: &Value, ty: &str, name: &str) -> Result<(), PromptTemplateError> {
    let bad = |detail: &str| {
        Err(PromptTemplateError::VariableValueType {
            name: name.to_string(),
            expected_type: ty.to_string(),
            detail: detail.to_string(),
        })
    };
    match ty {
        "str" => {
            if !v.is_string() {
                return bad("use a JSON string or prompt_variables_from_str_map");
            }
        }
        "int" => {
            if !json_is_integer(v) {
                return bad("use a JSON integer (not a float string)");
            }
        }
        "bool" => {
            if !v.is_boolean() {
                return bad("use JSON true/false");
            }
        }
        "list[int]" => {
            let Some(arr) = v.as_array() else {
                return bad("use a JSON array of integers");
            };
            for (i, el) in arr.iter().enumerate() {
                if !json_is_integer(el) {
                    return bad(&format!("element [{i}] is not an integer"));
                }
            }
        }
        "list[str]" => {
            let Some(arr) = v.as_array() else {
                return bad("use a JSON array of strings");
            };
            for (i, el) in arr.iter().enumerate() {
                if !el.is_string() {
                    return bad(&format!("element [{i}] is not a string"));
                }
            }
        }
        "dict[str, str]" => {
            let Some(obj) = v.as_object() else {
                return bad("use a JSON object with string keys and string values");
            };
            for (k, el) in obj {
                if !el.is_string() {
                    return bad(&format!("value for key '{k}' is not a string"));
                }
            }
        }
        _ => {
            return Err(PromptTemplateError::VariablesParse {
                prompt: name.to_string(),
                detail: format!("unknown declared type '{ty}'"),
            });
        }
    }
    Ok(())
}

fn validate_runtime_variables(
    prompt_name: &str,
    variables: &HashMap<String, Value>,
    specs: &[PromptVariable],
) -> Result<(), PromptTemplateError> {
    let allowed: HashSet<_> = specs.iter().map(|s| s.name.clone()).collect();
    let mut extra: Vec<_> = variables
        .keys()
        .filter(|k| !allowed.contains(*k))
        .cloned()
        .collect();
    if !extra.is_empty() {
        extra.sort();
        return Err(PromptTemplateError::VariableMismatch {
            prompt: prompt_name.to_string(),
            missing: vec![],
            extra,
        });
    }

    let mut missing = Vec::new();
    for spec in specs {
        match variables.get(&spec.name) {
            None => {
                if spec.required {
                    missing.push(spec.name.clone());
                }
            }
            Some(v) => validate_json_value_type(v, &spec.var_type, &spec.name)?,
        }
    }
    if !missing.is_empty() {
        missing.sort();
        return Err(PromptTemplateError::VariableMismatch {
            prompt: prompt_name.to_string(),
            missing,
            extra: vec![],
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
pub struct PromptVariable {
    pub name: String,
    pub var_type: String,
    pub required: bool,
}

impl PromptVariable {
    fn from_definition(def: VariableDef) -> Self {
        Self {
            name: def.name,
            var_type: def.var_type,
            required: def.required,
        }
    }
}

fn default_required_true() -> bool {
    true
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct VariablesFile {
    #[serde(default)]
    system_prompt: Vec<VariableDef>,
    #[serde(default)]
    user_prompt: Vec<VariableDef>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct VariableDef {
    name: String,
    #[serde(rename = "type")]
    var_type: String,
    #[serde(default = "default_required_true")]
    required: bool,
}

const ALLOWED_PROMPT_VAR_TYPES: &[&str] = &[
    "str",
    "int",
    "bool",
    "list[int]",
    "list[str]",
    "dict[str, str]",
];

#[derive(Clone)]
pub struct PromptTemplateData {
    pub name: String,
    pub template_text: String,
    pub variables: Vec<PromptVariable>,
}

fn prompt_dir(template: PromptTemplate) -> Result<&'static Dir<'static>, PromptTemplateError> {
    let path = format!("{}/{}", template.folder_path(), template.name());
    PROMPTS_DIR
        .get_dir(&path)
        .ok_or(PromptTemplateError::NotFound(path))
}

fn validate_var_type(var_type: &str) -> Result<(), PromptTemplateError> {
    if ALLOWED_PROMPT_VAR_TYPES.contains(&var_type) {
        Ok(())
    } else {
        Err(PromptTemplateError::VariablesParse {
            prompt: "variables.yml".into(),
            detail: format!("unsupported type '{var_type}'"),
        })
    }
}

fn load_variables_file(prompt_name: &str, dir: &Dir) -> Result<VariablesFile, PromptTemplateError> {
    let Some(file) = find_file_in_dir_by_name(dir, "variables.yml") else {
        return Ok(VariablesFile::default());
    };
    let contents = file
        .contents_utf8()
        .ok_or_else(|| PromptTemplateError::VariablesParse {
            prompt: prompt_name.to_string(),
            detail: "non-utf8 variables.yml".into(),
        })?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        return Ok(VariablesFile::default());
    }
    serde_yaml::from_str(trimmed).map_err(|err| PromptTemplateError::VariablesParse {
        prompt: prompt_name.to_string(),
        detail: err.to_string(),
    })
}

fn variable_defs_to_prompt_variables(
    defs: Vec<VariableDef>,
) -> Result<Vec<PromptVariable>, PromptTemplateError> {
    let mut seen = HashSet::new();
    let mut out = Vec::with_capacity(defs.len());
    for def in defs {
        validate_var_type(&def.var_type)?;
        if def.name.trim().is_empty() {
            return Err(PromptTemplateError::VariablesParse {
                prompt: "variables.yml".into(),
                detail: "variable name must be non-empty".into(),
            });
        }
        if !seen.insert(def.name.clone()) {
            return Err(PromptTemplateError::VariablesParse {
                prompt: "variables.yml".into(),
                detail: format!("duplicate variable '{}'", def.name),
            });
        }
        out.push(PromptVariable::from_definition(def));
    }
    Ok(out)
}

fn load_variables_for_template(
    prompt_name: &str,
    dir: &Dir,
    template: PromptTemplate,
) -> Result<Vec<PromptVariable>, PromptTemplateError> {
    let file = load_variables_file(prompt_name, dir)?;
    let defs = match template.template_file_name() {
        "system_prompt.j2" => file.system_prompt,
        "user_prompt.j2" => file.user_prompt,
        other => {
            return Err(PromptTemplateError::VariablesParse {
                prompt: prompt_name.to_string(),
                detail: format!("unexpected template file name {other}"),
            });
        }
    };
    variable_defs_to_prompt_variables(defs)
}

fn load_template_text(
    prompt_name: &str,
    template: PromptTemplate,
    dir: &Dir,
) -> Result<String, PromptTemplateError> {
    let file = find_file_in_dir_by_name(dir, template.template_file_name()).ok_or_else(|| {
        let available_files = dir
            .files()
            .map(|f| f.path().to_string_lossy().to_string())
            .collect::<Vec<_>>();
        PromptTemplateError::NotFound(format!(
            "{}/{} (available_files={available_files:?})",
            prompt_name,
            template.template_file_name()
        ))
    })?;
    file.contents_utf8()
        .map(|text| text.to_string())
        .ok_or_else(|| PromptTemplateError::Rendering {
            prompt: prompt_name.to_string(),
            detail: "unable to read template".into(),
        })
}

fn extract_template_variables(template: &str) -> HashSet<String> {
    let mut vars = HashSet::new();
    for cap in TEMPLATE_VAR_REGEX.captures_iter(template) {
        vars.insert(cap[1].to_string());
    }
    for cap in FOR_VAR_REGEX.captures_iter(template) {
        vars.insert(cap[1].to_string());
    }
    for cap in BLOCK_VAR_REGEX.captures_iter(template) {
        vars.insert(cap[1].to_string());
    }
    vars
}

fn ensure_variable_consistency(
    prompt_name: &str,
    defined: &HashSet<String>,
    referenced: &HashSet<String>,
) -> Result<(), PromptTemplateError> {
    let missing: Vec<_> = referenced.difference(defined).cloned().collect();
    let extra: Vec<_> = defined.difference(referenced).cloned().collect();
    if missing.is_empty() && extra.is_empty() {
        return Ok(());
    }
    Err(PromptTemplateError::VariableMismatch {
        prompt: prompt_name.to_string(),
        missing,
        extra,
    })
}

fn load_prompt(template: PromptTemplate) -> Result<PromptTemplateData, PromptTemplateError> {
    let prompt_name = template.name();
    let dir = prompt_dir(template)?;
    let template_text = load_template_text(prompt_name, template, dir)?;
    let yml_present = find_file_in_dir_by_name(dir, "variables.yml").is_some();
    let variables = load_variables_for_template(prompt_name, dir, template)?;
    let defined_names: HashSet<_> = variables.iter().map(|var| var.name.clone()).collect();
    let referenced = extract_template_variables(&template_text);
    if !referenced.is_empty() && !yml_present {
        return Err(PromptTemplateError::VariablesParse {
            prompt: prompt_name.to_string(),
            detail: "template uses variables but variables.yml is missing".into(),
        });
    }
    ensure_variable_consistency(prompt_name, &defined_names, &referenced)?;

    Ok(PromptTemplateData {
        name: prompt_name.into(),
        template_text,
        variables,
    })
}

fn find_file_in_dir_by_name<'a>(
    dir: &'a Dir,
    file_name: &str,
) -> Option<&'a include_dir::File<'a>> {
    dir.files().find(|f| {
        f.path()
            .file_name()
            .is_some_and(|name| name.to_string_lossy() == file_name)
    })
}

fn prompt_data(template: PromptTemplate) -> Result<PromptTemplateData, PromptTemplateError> {
    let mut cache = PROMPT_CACHE.lock().unwrap();
    if let Some(data) = cache.get(&template) {
        return Ok(data.clone());
    }
    let data = load_prompt(template)?;
    cache.insert(template, data.clone());
    Ok(data)
}

pub fn render_prompt(
    prompt: &PromptTemplateData,
    variables: &HashMap<String, Value>,
) -> Result<String, PromptTemplateError> {
    validate_runtime_variables(&prompt.name, variables, &prompt.variables)?;

    let mut context = Context::new();
    for (key, value) in variables {
        context.insert(key, value);
    }

    Tera::default()
        .render_str(&prompt.template_text, &context)
        .map_err(|err| PromptTemplateError::Rendering {
            prompt: prompt.name.clone(),
            detail: err.to_string(),
        })
}

pub struct PromptTemplateClient;

impl PromptTemplateClient {
    pub fn from_env() -> Result<Self, PromptTemplateError> {
        // No environment is required; this method exists for parity with the previous API.
        Ok(Self)
    }
}

/// Load and cache prompt metadata and template text (same as Python ``get_prompt_template``).
pub fn get_prompt_template(
    template: PromptTemplate,
) -> Result<PromptTemplateData, PromptTemplateError> {
    prompt_data(template)
}

/// Renders a prompt using variables from `variables.yml` for that template phase. Values must be
/// [`serde_json::Value`] variants matching the declared `type` (`str`, `int`, `bool`, `list[int]`,
/// `list[str]`, `dict[str, str]`). For all-string call sites, use [`prompt_variables_from_str_map`]. To load metadata
/// without rendering, use [`get_prompt_template`] (same as Python ``get_prompt_template``).
pub async fn get_and_render_prompt(
    _client: &PromptTemplateClient,
    template: PromptTemplate,
    variables: &HashMap<String, Value>,
    _label: Option<&str>,
) -> Result<String, PromptTemplateError> {
    let prompt = prompt_data(template)?;
    render_prompt(&prompt, variables)
}
