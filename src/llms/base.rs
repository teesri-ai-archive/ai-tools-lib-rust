use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
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
pub type ToolFuture = Pin<Box<dyn Future<Output = Result<Value, LlmError>> + Send>>;
pub type ToolExecutor = Arc<dyn Fn(Value) -> ToolFuture + Send + Sync>;

#[derive(Clone)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub executor: Option<ToolExecutor>,
}

impl std::fmt::Debug for Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("parameters", &self.parameters)
            .field("has_executor", &self.executor.is_some())
            .finish()
    }
}

impl Tool {
    pub fn new(name: String, description: String, parameters: Value) -> Self {
        Self {
            name,
            description,
            parameters,
            executor: None,
        }
    }

    pub fn with_executor(
        name: String,
        description: String,
        parameters: Value,
        executor: ToolExecutor,
    ) -> Self {
        Self {
            name,
            description,
            parameters,
            executor: Some(executor),
        }
    }

    pub async fn invoke(&self, args: Value) -> Result<Value, LlmError> {
        let executor = self.executor.as_ref().ok_or_else(|| {
            LlmError::Request(format!(
                "Tool '{}' was requested but has no executor",
                self.name
            ))
        })?;
        executor(args).await
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn tool_invoke_without_executor_returns_error() {
        let tool = Tool::new(
            "echo".to_string(),
            "Echo tool".to_string(),
            serde_json::json!({"type":"object"}),
        );
        let result = tool.invoke(serde_json::json!({"value":"x"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tool_invoke_with_executor_returns_value() {
        let tool = Tool::with_executor(
            "echo".to_string(),
            "Echo tool".to_string(),
            serde_json::json!({"type":"object"}),
            Arc::new(|args| Box::pin(async move { Ok(args) })),
        );
        let result = tool.invoke(serde_json::json!({"value":"ok"})).await.unwrap();
        assert_eq!(result, serde_json::json!({"value":"ok"}));
    }
}
