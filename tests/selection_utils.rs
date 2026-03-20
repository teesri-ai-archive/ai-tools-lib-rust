use ai_tools_lib_rust::llms::token_counter::TokenCounter;
use ai_tools_lib_rust::{
    llms::{
        BaseLLM, Message,
        base::{GenerateResponse, LlmError},
    },
    selection_utils::{PromptRenderer, SelectionError, choose_the_best_item_for_purpose_from_list},
};
use async_trait::async_trait;
use std::collections::HashMap;

struct MockLLM {
    logprobs: Vec<(String, f64)>,
}

#[async_trait]
impl BaseLLM for MockLLM {
    async fn generate(
        &self,
        _model: &str,
        _system_instruction: &str,
        _messages: &[Message],
        _token_counter: &TokenCounter,
        _return_logprobs: bool,
        _tools: Option<&[ai_tools_lib_rust::llms::base::Tool]>,
        _max_tokens: Option<u32>,
    ) -> Result<GenerateResponse, LlmError> {
        Ok(GenerateResponse::Logprobs(self.logprobs.clone()))
    }

    async fn generate_typed<T>(
        &self,
        _model: &str,
        _system_instruction: &str,
        _messages: &[Message],
        _response_type: T,
        _token_counter: &TokenCounter,
        _tools: Option<&[ai_tools_lib_rust::llms::base::Tool]>,
    ) -> Result<(T, Option<String>), LlmError>
    where
        T: serde::de::DeserializeOwned + Send + Sync,
    {
        Err(LlmError::Request("not implemented".to_string()))
    }

    async fn completion_call(
        &self,
        _messages: &[Message],
        _model: &str,
        _token_counter: Option<&TokenCounter>,
    ) -> Result<serde_json::Value, LlmError> {
        Err(LlmError::Request("not implemented".to_string()))
    }

    fn get_model(&self, _cheap: bool, _reasoning: bool, _multimodal: bool) -> String {
        "mock".into()
    }

    async fn generate_video_from_prompt_and_image(
        &self,
        _prompt: &str,
        _image_path: &str,
        _output_path: &str,
        _token_counter: &TokenCounter,
    ) -> Result<String, LlmError> {
        Err(LlmError::Request("not implemented".into()))
    }
}

struct MockRenderer;

#[async_trait]
impl PromptRenderer for MockRenderer {
    async fn render(
        &self,
        template: ai_tools_lib_rust::prompt_manager::PromptTemplate,
        variables: &HashMap<String, String>,
        _label: Option<&str>,
    ) -> Result<String, ai_tools_lib_rust::prompt_manager::prompt_layer::PromptLayerError> {
        let content = match template {
            ai_tools_lib_rust::prompt_manager::PromptTemplate::SelectionUtilsSystemPrompt => {
                "SYSTEM".to_string()
            }
            ai_tools_lib_rust::prompt_manager::PromptTemplate::SelectionUtilsUserPrompt => {
                variables.get("purpose").cloned().unwrap_or_default()
            }
            _ => "UNKNOWN".to_string(),
        };
        Ok(content)
    }
}

#[tokio::test]
async fn choose_best_item_returns_expected_label() {
    let llm = MockLLM {
        logprobs: vec![("A".into(), -1.0), ("B".into(), -0.1)],
    };
    let renderer = MockRenderer;
    let mut items = HashMap::new();
    items.insert("A".into(), "Apple".into());
    items.insert("B".into(), "Boat".into());

    let counter = TokenCounter::new();
    let result = choose_the_best_item_for_purpose_from_list(
        &items,
        "Need a vehicle",
        &counter,
        "vehicle",
        &llm,
        &renderer,
        None,
    )
    .await
    .expect("selection");
    assert_eq!(result.as_deref(), Some("B"));
}

#[tokio::test]
async fn choose_best_item_with_threshold_returns_none() {
    let llm = MockLLM {
        logprobs: vec![("A".into(), -10.0)],
    };
    let renderer = MockRenderer;
    let mut items = HashMap::new();
    items.insert("A".into(), "Apple".into());

    let counter = TokenCounter::new();
    let result = choose_the_best_item_for_purpose_from_list(
        &items,
        "Need food",
        &counter,
        "food",
        &llm,
        &renderer,
        Some(0.5),
    )
    .await
    .expect("selection");
    assert!(result.is_none());
}

#[tokio::test]
async fn choose_best_item_invalid_label_error() {
    let llm = MockLLM { logprobs: vec![] };
    let renderer = MockRenderer;
    let mut items = HashMap::new();
    items.insert("AB".into(), "Bad".into());

    let counter = TokenCounter::new();
    let err = choose_the_best_item_for_purpose_from_list(
        &items,
        "Need food",
        &counter,
        "food",
        &llm,
        &renderer,
        None,
    )
    .await
    .err()
    .unwrap();
    assert!(matches!(err, SelectionError::InvalidLabel(_)));
}

#[tokio::test]
async fn choose_best_item_empty_items_error() {
    let llm = MockLLM { logprobs: vec![] };
    let renderer = MockRenderer;
    let items = HashMap::new();

    let counter = TokenCounter::new();
    let err = choose_the_best_item_for_purpose_from_list(
        &items,
        "Need food",
        &counter,
        "food",
        &llm,
        &renderer,
        None,
    )
    .await
    .err()
    .unwrap();
    assert!(matches!(err, SelectionError::InvalidInput(_)));
}
