use crate::prompt_manager::templates::PromptTemplate;
use log::{debug, error, info, warn};
use once_cell::sync::Lazy;
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::env;
use thiserror::Error;

static FSTRING_PLACEHOLDER: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"\{([a-zA-Z0-9_]+)\}").unwrap());

#[derive(Debug, Deserialize)]
pub struct PromptLayerTemplate {
    pub prompt_name: String,
    pub prompt_template: PromptTemplatePayload,
}

#[derive(Debug, Deserialize)]
pub struct PromptTemplatePayload {
    pub messages: Vec<PromptMessage>,
}

#[derive(Debug, Deserialize)]
pub struct PromptMessage {
    pub input_variables: Vec<String>,
    pub template_format: String,
    pub content: Vec<PromptContent>,
}

#[derive(Debug, Deserialize)]
pub struct PromptContent {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub annotations: Option<Value>,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PromptTemplateData {
    pub name: String,
    pub input_variables: Vec<String>,
    pub template_format: String,
    pub template_text: String,
}

/// Errors surfaced by the prompt manager.
#[derive(Debug, Error)]
pub enum PromptLayerError {
    #[error("missing PROMPTLAYER_API_KEY environment variable")]
    MissingApiKey,
    #[error("failed to fetch template: {0}")]
    Http(#[from] reqwest::Error),
    #[error("template parsing failed: {0}")]
    Parse(String),
    #[error("template not found: {0}")]
    NotFound(String),
}

/// Client against the PromptLayer API.
pub struct PromptLayerClient {
    client: Client,
    api_key: String,
    base_url: String,
}

impl PromptLayerClient {
    pub fn from_env() -> Result<Self, PromptLayerError> {
        let api_key =
            env::var("PROMPTLAYER_API_KEY").map_err(|_| PromptLayerError::MissingApiKey)?;
        let base_url = env::var("PROMPTLAYER_BASE_URL")
            .unwrap_or_else(|_| "https://api.promptlayer.com".into());
        Ok(Self {
            client: Client::new(),
            api_key,
            base_url,
        })
    }

    pub async fn get_template(
        &self,
        template: PromptTemplate,
        label: Option<&str>,
    ) -> Result<PromptLayerTemplate, PromptLayerError> {
        let environment = env::var("ENV").unwrap_or_else(|_| "prod".into());
        let folder_path = template.folder_path();
        let label = label.or(Some(environment.as_str()));

        info!("Attempting to retrieve template {}", template.name());

        if let Some(template_data) = self.fetch_template(template.name(), None, None).await? {
            info!("Template {} found without label", template_data.prompt_name);
            return Ok(template_data);
        }

        if let Some(label) = label
            && let Some(template_data) = self
                .fetch_template(template.name(), Some(label), None)
                .await?
        {
            info!("Template {} found with label {}", template.name(), label);
            return Ok(template_data);
        }

        if let Some(label) = label
            && let Some(template_data) = self
                .fetch_template(template.name(), Some(label), Some(folder_path))
                .await?
        {
            info!(
                "Template {} found with label {} and folder {}",
                template.name(),
                label,
                folder_path
            );
            return Ok(template_data);
        }

        error!("Template {} not found", template.name());
        Err(PromptLayerError::NotFound(template.name().to_string()))
    }

    async fn fetch_template(
        &self,
        name: &str,
        label: Option<&str>,
        folder_path: Option<&str>,
    ) -> Result<Option<PromptLayerTemplate>, PromptLayerError> {
        let url = format!("{}/templates/get", self.base_url);
        let mut request = self.client.get(&url);

        let mut params = vec![("template_name", name)];
        if let Some(label) = label {
            params.push(("label", label));
        }
        if let Some(folder_path) = folder_path {
            params.push(("folder_path", folder_path));
        }

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", self.api_key).parse().unwrap(),
        );
        request = request.headers(headers).query(&params);

        debug!("Fetching PromptLayer template {} {:?}", name, params);
        let response = request.send().await?;
        if response.status().is_client_error() {
            warn!(
                "Template {} could not be located (status={})",
                name,
                response.status()
            );
            return Ok(None);
        }

        let template = response.json::<PromptLayerTemplate>().await?;
        Ok(Some(template))
    }
}

pub fn extract_prompt(
    template: &PromptLayerTemplate,
) -> Result<PromptTemplateData, PromptLayerError> {
    let message = template
        .prompt_template
        .messages
        .first()
        .ok_or_else(|| PromptLayerError::Parse("messages missing".into()))?;

    let content = message
        .content
        .first()
        .ok_or_else(|| PromptLayerError::Parse("content missing".into()))?;

    let text = content
        .text
        .clone()
        .ok_or_else(|| PromptLayerError::Parse("text missing".into()))?;

    Ok(PromptTemplateData {
        name: template.prompt_name.clone(),
        input_variables: message.input_variables.clone(),
        template_format: message.template_format.clone(),
        template_text: text,
    })
}

pub fn validate_variables(
    prompt: &PromptTemplateData,
    variables: &HashMap<String, String>,
) -> Result<(), PromptLayerError> {
    let missing: Vec<_> = prompt
        .input_variables
        .iter()
        .filter(|key| !variables.contains_key(*key))
        .cloned()
        .collect();

    if !missing.is_empty() {
        return Err(PromptLayerError::Parse(format!(
            "Missing variables for prompt '{}': {:?}",
            prompt.name, missing
        )));
    }

    Ok(())
}

pub fn render_f_string(
    prompt: &PromptTemplateData,
    variables: &HashMap<String, String>,
) -> Result<String, PromptLayerError> {
    let mut result = prompt.template_text.clone();
    for var in &prompt.input_variables {
        let placeholder = format!("{{{}}}", var);
        let value = variables.get(var).ok_or_else(|| {
            PromptLayerError::Parse(format!("Missing variable in f-string: {}", var))
        })?;
        result = result.replace(&placeholder, value);
    }

    if FSTRING_PLACEHOLDER.is_match(&result) {
        return Err(PromptLayerError::Parse(
            "f-string contains unresolved placeholders".into(),
        ));
    }

    Ok(result)
}

pub fn render_jinja(
    prompt: &PromptTemplateData,
    variables: &HashMap<String, String>,
) -> Result<String, PromptLayerError> {
    let mut tera = tera::Tera::default();
    tera.add_raw_template("prompt", &prompt.template_text)
        .map_err(|error: tera::Error| PromptLayerError::Parse(error.to_string()))?;

    let mut context = tera::Context::new();
    for (key, value) in variables {
        context.insert(key, value);
    }

    tera.render("prompt", &context)
        .map_err(|error| PromptLayerError::Parse(error.to_string()))
}

pub fn render_prompt(
    prompt: &PromptTemplateData,
    variables: &HashMap<String, String>,
) -> Result<String, PromptLayerError> {
    validate_variables(prompt, variables)?;
    match prompt.template_format.as_str() {
        "jinja2" => render_jinja(prompt, variables),
        "f-string" => render_f_string(prompt, variables),
        other => Err(PromptLayerError::Parse(format!(
            "Unsupported template format: {}",
            other
        ))),
    }
}

pub async fn get_and_render_prompt(
    client: &PromptLayerClient,
    template: PromptTemplate,
    variables: &HashMap<String, String>,
    label: Option<&str>,
) -> Result<String, PromptLayerError> {
    let template_dict = client.get_template(template, label).await?;
    let prompt = extract_prompt(&template_dict)?;
    render_prompt(&prompt, variables)
}
