use super::base::{BaseLLM, GenerateResponse, LlmError, Message, Tool};
use super::token_counter::TokenCounter;
use async_trait::async_trait;
use log::{debug, info};

/// Configuration mirrors python defaults.
pub struct GeminiConfig {
    pub reasoning_model: String,
    pub multimodal_model: String,
    pub cheap_model: String,
    pub default_model: String,
}

impl Default for GeminiConfig {
    fn default() -> Self {
        Self {
            reasoning_model: "gemini-3-pro-preview".to_string(),
            multimodal_model: "gemini-2.5-flash".to_string(),
            cheap_model: "gemini-2.5-flash-lite".to_string(),
            default_model: "gemini-2.5-flash".to_string(),
        }
    }
}

/// Gemini wrapper (stubbed).
pub struct GeminiLLM {
    _api_key: String,
    config: GeminiConfig,
}

impl GeminiLLM {
    pub fn new(api_key: String, config: Option<GeminiConfig>) -> Self {
        Self {
            _api_key: api_key,
            config: config.unwrap_or_default(),
        }
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
            "Gemini generation not implemented in Rust placeholder".to_string(),
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
            "Gemini typed generation not implemented".to_string(),
        ))
    }

    async fn completion_call(
        &self,
        _messages: &[Message],
        _model: &str,
        _token_counter: Option<&TokenCounter>,
    ) -> Result<serde_json::Value, LlmError> {
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
