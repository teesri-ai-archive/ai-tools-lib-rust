use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use thiserror::Error;

/// Role of a chat message.
#[derive(Clone, Debug)]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Simplified message struct that mirrors the Python representation.
#[derive(Clone, Debug)]
pub struct Message {
    pub role: Role,
    pub content: Value,
}

impl Message {
    /// Convenience constructor for text-only content.
    pub fn new(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: Value::String(text.into()),
        }
    }
}

/// Tool metadata consumed by the LLM wrappers.
#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// Response returned by the generic `generate()` method.
#[derive(Clone, Debug)]
pub enum GenerateResponse {
    Text(String),
    Logprobs(Vec<(String, f64)>),
}

/// Base LLM error wrapper.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM request failed: {0}")]
    Request(String),
    #[error("Logprobs unavailable")]
    MissingLogprobs,
    #[error("Invalid response format")]
    InvalidResponse,
}

/// Trait with shared LLM behavior.
#[async_trait]
pub trait BaseLLM: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    async fn generate(
        &self,
        model: &str,
        system_instruction: &str,
        messages: &[Message],
        token_counter: &super::token_counter::TokenCounter,
        return_logprobs: bool,
        tools: Option<&[Tool]>,
        max_tokens: Option<u32>,
    ) -> Result<GenerateResponse, LlmError>;

    async fn generate_typed<T>(
        &self,
        model: &str,
        system_instruction: &str,
        messages: &[Message],
        response_type: T,
        token_counter: &super::token_counter::TokenCounter,
        tools: Option<&[Tool]>,
    ) -> Result<(T, Option<String>), LlmError>
    where
        T: DeserializeOwned + Send + Sync;

    async fn completion_call(
        &self,
        messages: &[Message],
        model: &str,
        token_counter: Option<&super::token_counter::TokenCounter>,
    ) -> Result<Value, LlmError>;

    fn get_model(&self, cheap: bool, reasoning: bool, multimodal: bool) -> String;

    async fn generate_video_from_prompt_and_image(
        &self,
        prompt: &str,
        image_path: &str,
        output_path: &str,
        token_counter: &super::token_counter::TokenCounter,
    ) -> Result<String, LlmError>;
}
