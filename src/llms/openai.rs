use super::base::{BaseLLM, GenerateResponse, LlmError, Message, Tool};
use super::token_counter::TokenCounter;
use async_trait::async_trait;
use log::{debug, info};

/// OpenAI wrapper mimicking the Python interface.
pub struct OpenAILLM {
    _api_key: String,
}

impl OpenAILLM {
    pub fn new(api_key: String) -> Self {
        Self { _api_key: api_key }
    }
}

#[async_trait]
impl BaseLLM for OpenAILLM {
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
        debug!("OpenAILLM.generate called but not implemented");
        Err(LlmError::Request(
            "OpenAI generation not configured in Rust implementation".to_string(),
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
            "OpenAI typed generation not implemented".to_string(),
        ))
    }

    async fn completion_call(
        &self,
        _messages: &[Message],
        _model: &str,
        _token_counter: Option<&TokenCounter>,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::Request(
            "OpenAI completion call not implemented".to_string(),
        ))
    }

    fn get_model(&self, _cheap: bool, _reasoning: bool, _multimodal: bool) -> String {
        "gpt-5-nano".to_string()
    }

    async fn generate_video_from_prompt_and_image(
        &self,
        _prompt: &str,
        _image_path: &str,
        _output_path: &str,
        _token_counter: &TokenCounter,
    ) -> Result<String, LlmError> {
        info!("OpenAI video generation is not supported; raise NotImplementedError");
        Err(LlmError::Request(
            "OpenAI does not support video generation".to_string(),
        ))
    }
}
