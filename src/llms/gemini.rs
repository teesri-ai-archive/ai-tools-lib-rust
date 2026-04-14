use super::base::{BaseLLM, GenerateResponse, LlmError, Message, Tool};
use super::token_counter::TokenCounter;
use async_trait::async_trait;
use log::{debug, info, warn};
use reqwest::Client;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::time::Duration;

/// Configuration mirrors python defaults.
pub struct GeminiConfig {
    pub reasoning_model: String,
    pub multimodal_model: String,
    pub cheap_model: String,
    pub default_model: String,
    /// Per-request HTTP timeout for the Gemini REST API (seconds). Large JSON / reasoning
    /// responses can exceed short client defaults and fail with read timeouts.
    pub http_timeout_seconds: u64,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            reasoning_model: "gemini-3-pro-preview".to_string(),
            multimodal_model: "gemini-2.5-flash".to_string(),
            cheap_model: "gemini-2.5-flash-lite".to_string(),
            default_model: "gemini-2.5-flash".to_string(),
            http_timeout_seconds: 600,
        }
    }
}

/// Gemini REST client (generateContent with JSON schema).
pub struct GeminiLLM {
    api_key: String,
    config: GeminiConfig,
    http: Client,
}

impl GeminiLLM {
    pub fn new(api_key: String, config: Option<GeminiConfig>) -> Self {
        let config = config.unwrap_or_default();
        let timeout = Duration::from_secs(config.http_timeout_seconds);
        let http = Client::builder()
            .timeout(timeout)
            .build()
            .expect("reqwest Client builder should succeed");
        Self {
            api_key,
            config,
            http,
        }
    }

    /// Structured JSON output using `generationConfig.responseSchema` (Gemini 2.x).
    pub async fn generate_structured_json<T: DeserializeOwned>(
        &self,
        model: &str,
        system_instruction: &str,
        user_text: &str,
        response_schema: Value,
        token_counter: &TokenCounter,
    ) -> Result<T, LlmError> {
        let schema = strip_additional_properties(response_schema);
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        );
        let body = serde_json::json!({
            "systemInstruction": {
                "parts": [{ "text": system_instruction }]
            },
            "contents": [{
                "role": "user",
                "parts": [{ "text": user_text }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": schema,
            }
        });

        log_gemini_request(system_instruction, Some(user_text), &body);

        let resp = self
            .http
            .post(&url)
            .query(&[("key", self.api_key.as_str())])
            .json(&body)
            .send()
            .await
            .map_err(|e| LlmError::Request(format!("Gemini HTTP error: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| LlmError::Request(format!("Gemini read body: {e}")))?;

        if !status.is_success() {
            return Err(LlmError::Request(format!(
                "Gemini API status {status}: {text}"
            )));
        }

        let parsed: GenerateContentResponse = serde_json::from_str(&text).map_err(|e| {
            LlmError::Request(format!("Gemini JSON parse error: {e}; body: {text}"))
        })?;

        log_gemini_response(&parsed);

        if let Some(ref err) = parsed.error {
            return Err(LlmError::Request(format!(
                "{}: {}",
                err.status.as_deref().unwrap_or(""),
                err.message.as_deref().unwrap_or("")
            )));
        }

        if let Some(ref meta) = parsed.usage_metadata {
            token_counter.add_counts(
                model,
                meta.prompt_token_count.unwrap_or(0),
                meta.candidates_token_count.unwrap_or(0),
                meta.thoughts_token_count.unwrap_or(0),
            );
        }

        let json_text =
            extract_last_non_thought_json_text(&parsed).ok_or(LlmError::InvalidResponse)?;

        serde_json::from_str::<T>(&json_text).map_err(|e| {
            LlmError::Request(format!("Gemini output JSON decode: {e}; text: {json_text}"))
        })
    }
}

fn log_gemini_request(system_instruction: &str, user_messages: Option<&str>, api_body: &Value) {
    info!(
        "Gemini request — system instruction:\n{}",
        if system_instruction.is_empty() {
            "(none)"
        } else {
            system_instruction
        }
    );
    if let Some(msgs) = user_messages {
        info!("Gemini request — user messages:\n{msgs}");
    }
    let contents_str =
        serde_json::to_string_pretty(api_body).unwrap_or_else(|_| api_body.to_string());
    info!("Gemini request — API contents:\n{contents_str}");
}

fn log_gemini_response(parsed: &GenerateContentResponse) {
    match parsed.candidates.as_ref() {
        None => {
            info!("Gemini response — text: {parsed:?}");
        }
        Some(c) if c.is_empty() => {
            info!("Gemini response — text: {parsed:?}");
        }
        Some(cands) => {
            let c0 = &cands[0];
            let parts_slice: &[Part] = c0
                .content
                .as_ref()
                .and_then(|c| c.parts.as_deref())
                .unwrap_or(&[]);
            if parts_slice.is_empty() {
                warn!(
                    "Gemini response had no content.parts: finish_reason={:?} prompt_feedback={:?}",
                    c0.finish_reason, parsed.prompt_feedback
                );
            }
            for part in parts_slice {
                let Some(t) = part.text.as_ref() else {
                    continue;
                };
                if part.thought == Some(true) {
                    info!("Gemini response — thought: {t}");
                } else {
                    info!("Gemini response — text: {t}");
                }
            }
        }
    }
}

/// Last text part that is not a reasoning "thought" segment (models may emit thought parts first).
fn extract_last_non_thought_json_text(parsed: &GenerateContentResponse) -> Option<String> {
    let parts = parsed
        .candidates
        .as_ref()?
        .first()?
        .content
        .as_ref()?
        .parts
        .as_ref()?;
    let mut last: Option<String> = None;
    for p in parts {
        if let Some(t) = p.text.as_ref()
            && p.thought != Some(true)
        {
            last = Some(t.clone());
        }
    }
    last
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
    #[serde(rename = "promptFeedback")]
    prompt_feedback: Option<Value>,
    error: Option<GeminiRestError>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<Content>,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Content {
    parts: Option<Vec<Part>>,
}

#[derive(Debug, Deserialize)]
struct Part {
    text: Option<String>,
    #[serde(default)]
    thought: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct UsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
    #[serde(rename = "thoughtsTokenCount")]
    thoughts_token_count: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct GeminiRestError {
    status: Option<String>,
    message: Option<String>,
}

fn strip_additional_properties(v: Value) -> Value {
    match v {
        Value::Object(mut map) => {
            map.remove("additionalProperties");
            let keys: Vec<String> = map.keys().cloned().collect();
            for k in keys {
                if let Some(inner) = map.get_mut(&k) {
                    *inner = strip_additional_properties(inner.clone());
                }
            }
            Value::Object(map)
        }
        Value::Array(items) => {
            Value::Array(items.into_iter().map(strip_additional_properties).collect())
        }
        other => other,
    }
}

#[async_trait]
impl BaseLLM for GeminiLLM {
    async fn generate(
        &self,
        _model: &str,
        _system_instruction: &str,
        _messages: &[Message],
        _token_counter: &TokenCounter,
        _return_logprobs: bool,
        _tools: Option<&[Tool]>,
        _max_tokens: Option<u32>,
    ) -> Result<GenerateResponse, LlmError> {
        debug!("GeminiLLM.generate invoked but not implemented");
        Err(LlmError::Request(
            "Gemini free-form generate not implemented in Rust".to_string(),
        ))
    }

    async fn generate_typed<T>(
        &self,
        _model: &str,
        _system_instruction: &str,
        _messages: &[Message],
        _response_type: T,
        _token_counter: &TokenCounter,
        _tools: Option<&[Tool]>,
    ) -> Result<(T, Option<String>), LlmError>
    where
        T: serde::de::DeserializeOwned + Send + Sync,
    {
        Err(LlmError::Request(
            "Use GeminiLLM::generate_structured_json with an explicit responseSchema".to_string(),
        ))
    }

    async fn completion_call(
        &self,
        _messages: &[Message],
        _model: &str,
        _token_counter: Option<&TokenCounter>,
    ) -> Result<Value, LlmError> {
        Err(LlmError::Request(
            "Gemini completion call not implemented".to_string(),
        ))
    }

    fn get_model(&self, cheap: bool, reasoning: bool, multimodal: bool) -> String {
        if reasoning {
            return self.config.reasoning_model.clone();
        }
        if multimodal {
            return self.config.multimodal_model.clone();
        }
        if cheap {
            return self.config.cheap_model.clone();
        }
        self.config.default_model.clone()
    }

    async fn generate_video_from_prompt_and_image(
        &self,
        _prompt: &str,
        _image_path: &str,
        _output_path: &str,
        _token_counter: &TokenCounter,
    ) -> Result<String, LlmError> {
        info!("Gemini video generation is not implemented in Rust version");
        Err(LlmError::Request(
            "Gemini video generation not implemented".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_additional_removes_nested() {
        let v = serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "a": { "type": "string", "additionalProperties": true }
            }
        });
        let out = strip_additional_properties(v);
        let obj = out.as_object().unwrap();
        assert!(!obj.contains_key("additionalProperties"));
        let a = obj.get("properties").unwrap()["a"].as_object().unwrap();
        assert!(!a.contains_key("additionalProperties"));
    }
}
