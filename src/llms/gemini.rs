use super::base::{BaseLLM, GenerateResponse, LlmError, Message, Tool};
use super::token_counter::TokenCounter;
use async_trait::async_trait;
use futures::future::join_all;
use log::{debug, info};
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

        if let Some(err) = parsed.error {
            return Err(LlmError::Request(format!(
                "{}: {}",
                err.status.unwrap_or_default(),
                err.message.unwrap_or_default()
            )));
        }

        if let Some(meta) = parsed.usage_metadata {
            token_counter.add_counts(
                model,
                meta.prompt_token_count.unwrap_or(0),
                meta.candidates_token_count.unwrap_or(0),
                meta.thoughts_token_count.unwrap_or(0),
            );
        }

        let json_text = parsed
            .candidates
            .and_then(|c| c.into_iter().next())
            .and_then(|c| c.content)
            .and_then(|content| content.parts)
            .and_then(|parts| parts.into_iter().find_map(|p| p.text))
            .ok_or(LlmError::InvalidResponse)?;

        serde_json::from_str::<T>(&json_text).map_err(|e| {
            LlmError::Request(format!("Gemini output JSON decode: {e}; text: {json_text}"))
        })
    }

    fn message_parts(content: &Value) -> Result<Vec<Value>, LlmError> {
        match content {
            Value::String(text) => Ok(vec![serde_json::json!({ "text": text })]),
            Value::Object(_) => Ok(vec![content.clone()]),
            Value::Array(items) => {
                let mut parts = Vec::with_capacity(items.len());
                for item in items {
                    match item {
                        Value::String(text) => parts.push(serde_json::json!({ "text": text })),
                        Value::Object(_) => parts.push(item.clone()),
                        other => {
                            return Err(LlmError::Request(format!(
                                "Unsupported message part content: {other}"
                            )));
                        }
                    }
                }
                Ok(parts)
            }
            other => Err(LlmError::Request(format!(
                "Unsupported message content type: {other}"
            ))),
        }
    }

    fn messages_to_contents(messages: &[Message]) -> Result<Vec<Value>, LlmError> {
        messages
            .iter()
            .map(|msg| {
                let role = match msg.role {
                    super::base::Role::Assistant => "model",
                    _ => "user",
                };
                Ok(serde_json::json!({
                    "role": role,
                    "parts": Self::message_parts(&msg.content)?,
                }))
            })
            .collect()
    }

    fn build_tools_payload(tools: &[Tool]) -> Value {
        let function_declarations: Vec<Value> = tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect();
        serde_json::json!([{
            "functionDeclarations": function_declarations
        }])
    }

    fn extract_parts(response: &Value) -> Vec<Value> {
        response
            .get("candidates")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.as_array())
            .cloned()
            .unwrap_or_default()
    }

    fn extract_first_text(parts: &[Value]) -> Option<String> {
        parts.iter()
            .find_map(|part| part.get("text").and_then(|t| t.as_str()).map(str::to_string))
    }

    fn update_token_counter_from_response(
        response: &Value,
        model: &str,
        token_counter: &TokenCounter,
    ) {
        if let Some(meta) = response.get("usageMetadata") {
            token_counter.add_counts(
                model,
                meta.get("promptTokenCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                meta.get("candidatesTokenCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
                meta.get("thoughtsTokenCount")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
            );
        }
    }

    fn collect_function_calls(parts: &[Value]) -> Vec<PendingFunctionCall> {
        let mut calls = Vec::new();
        for part in parts {
            let Some(fc) = part.get("functionCall") else {
                continue;
            };
            let Some(name) = fc.get("name").and_then(|n| n.as_str()) else {
                continue;
            };
            let id = fc
                .get("id")
                .and_then(|i| i.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            let args = fc
                .get("args")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({}));
            let mut raw_part = part.clone();
            if raw_part
                .get("functionCall")
                .and_then(|v| v.get("id"))
                .is_none()
            {
                raw_part["functionCall"]["id"] = Value::String(id.clone());
            }
            calls.push(PendingFunctionCall {
                name: name.to_string(),
                args,
                raw_part,
            });
        }
        calls
    }

    async fn execute_function_calls_parallel(
        calls: &[PendingFunctionCall],
        tools: &[Tool],
    ) -> Result<Vec<Value>, LlmError> {
        let futures = calls.iter().map(|call| async move {
            let tool = tools
                .iter()
                .find(|t| t.name == call.name)
                .ok_or_else(|| {
                    LlmError::Request(format!(
                        "Model requested unknown tool '{}'",
                        call.name
                    ))
                })?;
            tool.invoke(call.args.clone()).await
        });
        let results = join_all(futures).await;
        results.into_iter().collect()
    }

    fn build_function_response_parts(calls: &[PendingFunctionCall], results: &[Value]) -> Vec<Value> {
        calls
            .iter()
            .zip(results.iter())
            .map(|(call, result)| {
                serde_json::json!({
                    "functionResponse": {
                        "name": call.name,
                        "response": { "result": result }
                    }
                })
            })
            .collect()
    }
}

#[derive(Clone, Debug)]
struct PendingFunctionCall {
    name: String,
    args: Value,
    raw_part: Value,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Option<Vec<Candidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<UsageMetadata>,
    error: Option<GeminiRestError>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Option<Content>,
}

#[derive(Debug, Deserialize)]
struct Content {
    parts: Option<Vec<Part>>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct Part {
    text: Option<String>,
    #[serde(rename = "functionCall")]
    function_call: Option<FunctionCall>,
    #[serde(rename = "thoughtSignature")]
    thought_signature: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct FunctionCall {
    name: Option<String>,
    id: Option<String>,
    args: Option<Value>,
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
        model: &str,
        system_instruction: &str,
        messages: &[Message],
        token_counter: &TokenCounter,
        return_logprobs: bool,
        tools: Option<&[Tool]>,
        max_tokens: Option<u32>,
    ) -> Result<GenerateResponse, LlmError> {
        if return_logprobs {
            return Err(LlmError::MissingLogprobs);
        }

        let mut contents = Self::messages_to_contents(messages)?;
        let url =
            format!("https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent");
        let mut generation_config = serde_json::Map::new();
        if let Some(max_tokens) = max_tokens {
            generation_config.insert("maxOutputTokens".to_string(), Value::from(max_tokens));
        }

        loop {
            let mut body = serde_json::json!({
                "systemInstruction": { "parts": [{ "text": system_instruction }] },
                "contents": contents,
            });
            if !generation_config.is_empty() {
                body["generationConfig"] = Value::Object(generation_config.clone());
            }
            if let Some(tools) = tools {
                body["tools"] = Self::build_tools_payload(tools);
                body["toolConfig"] = serde_json::json!({
                    "functionCallingConfig": { "mode": "AUTO" }
                });
            }

            let resp = self
                .http
                .post(&url)
                .query(&[("key", self.api_key.as_str())])
                .json(&body)
                .send()
                .await
                .map_err(|e| LlmError::Request(format!("Gemini HTTP error: {e}")))?;
            let status = resp.status();
            let response_text = resp
                .text()
                .await
                .map_err(|e| LlmError::Request(format!("Gemini read body: {e}")))?;
            if !status.is_success() {
                return Err(LlmError::Request(format!(
                    "Gemini API status {status}: {response_text}"
                )));
            }

            let response: Value = serde_json::from_str(&response_text).map_err(|e| {
                LlmError::Request(format!("Gemini JSON parse error: {e}; body: {response_text}"))
            })?;
            if let Some(err) = response.get("error") {
                return Err(LlmError::Request(format!("Gemini API error: {err}")));
            }

            Self::update_token_counter_from_response(&response, model, token_counter);

            let parts = Self::extract_parts(&response);
            let function_calls = Self::collect_function_calls(&parts);
            if function_calls.is_empty() || tools.is_none() {
                let text = Self::extract_first_text(&parts).ok_or(LlmError::InvalidResponse)?;
                return Ok(GenerateResponse::Text(text));
            }

            let tools = tools.expect("tools checked above");
            let results = Self::execute_function_calls_parallel(&function_calls, tools).await?;

            let model_parts: Vec<Value> = function_calls.iter().map(|c| c.raw_part.clone()).collect();
            let response_parts = Self::build_function_response_parts(&function_calls, &results);
            contents.push(serde_json::json!({ "role": "model", "parts": model_parts }));
            contents.push(serde_json::json!({ "role": "user", "parts": response_parts }));
            debug!(
                "Processed {} Gemini function calls in parallel",
                function_calls.len()
            );
        }
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

    #[test]
    fn response_part_deserializes_function_call_and_thought_signature() {
        let payload = serde_json::json!({
            "text": null,
            "functionCall": {
                "name": "get_full_transcript",
                "id": "abc123",
                "args": {}
            },
            "thoughtSignature": "Zm9vYmFy"
        });
        let part: Part = serde_json::from_value(payload).unwrap();
        assert!(part.text.is_none());
        assert!(part.function_call.is_some());
        assert_eq!(
            part.function_call.as_ref().and_then(|fc| fc.name.as_deref()),
            Some("get_full_transcript")
        );
        assert_eq!(
            part.function_call.as_ref().and_then(|fc| fc.id.as_deref()),
            Some("abc123")
        );
        assert_eq!(part.thought_signature.as_deref(), Some("Zm9vYmFy"));
    }

    #[test]
    fn first_text_part_is_selected_even_after_function_call_part() {
        let content = Content {
            parts: Some(vec![
                Part {
                    text: None,
                    function_call: Some(FunctionCall {
                        name: Some("get_full_transcript".to_string()),
                        id: Some("abc123".to_string()),
                        args: Some(serde_json::json!({})),
                    }),
                    thought_signature: Some("sig".to_string()),
                },
                Part {
                    text: Some(r#"{"answer":"ok"}"#.to_string()),
                    function_call: None,
                    thought_signature: None,
                },
            ]),
        };

        let extracted = content
            .parts
            .unwrap()
            .into_iter()
            .find_map(|p| p.text)
            .unwrap();
        assert_eq!(extracted, r#"{"answer":"ok"}"#);
    }

    #[test]
    fn collect_function_calls_assigns_id_and_preserves_order() {
        let parts = vec![
            serde_json::json!({
                "functionCall": { "name": "a_tool", "args": {"x":1} },
                "thoughtSignature": "sig-a"
            }),
            serde_json::json!({
                "functionCall": { "id": "call-2", "name": "b_tool", "args": {"y":2} }
            }),
        ];
        let calls = GeminiLLM::collect_function_calls(&parts);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a_tool");
        assert_eq!(calls[1].name, "b_tool");
        assert_eq!(
            calls[1].raw_part["functionCall"]["id"].as_str(),
            Some("call-2")
        );
        assert!(calls[0].raw_part["functionCall"]["id"].as_str().is_some());
    }

    #[tokio::test]
    async fn execute_function_calls_parallel_returns_results_in_call_order() {
        use std::sync::Arc;
        use tokio::time::{Duration, sleep};

        let tools = vec![
            Tool::with_executor(
                "slow".to_string(),
                "Slow tool".to_string(),
                serde_json::json!({"type":"object"}),
                Arc::new(|args| {
                    Box::pin(async move {
                        sleep(Duration::from_millis(40)).await;
                        Ok(serde_json::json!({"tool":"slow","args":args}))
                    })
                }),
            ),
            Tool::with_executor(
                "fast".to_string(),
                "Fast tool".to_string(),
                serde_json::json!({"type":"object"}),
                Arc::new(|args| {
                    Box::pin(async move {
                        sleep(Duration::from_millis(5)).await;
                        Ok(serde_json::json!({"tool":"fast","args":args}))
                    })
                }),
            ),
        ];
        let calls = vec![
            PendingFunctionCall {
                name: "slow".to_string(),
                args: serde_json::json!({"value":"first"}),
                raw_part: serde_json::json!({}),
            },
            PendingFunctionCall {
                name: "fast".to_string(),
                args: serde_json::json!({"value":"second"}),
                raw_part: serde_json::json!({}),
            },
        ];

        let results = GeminiLLM::execute_function_calls_parallel(&calls, &tools)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0]["tool"].as_str(), Some("slow"));
        assert_eq!(results[1]["tool"].as_str(), Some("fast"));
    }
}
