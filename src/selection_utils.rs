use crate::llms::base::{GenerateResponse, LlmError, Role};
use crate::llms::{BaseLLM, Message, TokenCounter};
use crate::prompt_manager::{PromptTemplate, prompt_layer::PromptLayerError};
use async_trait::async_trait;
use log::info;
use std::collections::HashMap;

#[derive(Debug)]
pub enum SelectionError {
    InvalidLabel(String),
    InvalidInput(String),
    InvalidLLMResponse,
    LlmError(LlmError),
    PromptLayer(PromptLayerError),
}

impl From<LlmError> for SelectionError {
    fn from(error: LlmError) -> Self {
        SelectionError::LlmError(error)
    }
}

impl From<PromptLayerError> for SelectionError {
    fn from(error: PromptLayerError) -> Self {
        SelectionError::PromptLayer(error)
    }
}

#[async_trait]
pub trait PromptRenderer: Send + Sync {
    async fn render(
        &self,
        template: PromptTemplate,
        variables: &HashMap<String, String>,
        label: Option<&str>,
    ) -> Result<String, crate::prompt_manager::prompt_layer::PromptLayerError>;
}

pub struct PromptLayerRenderer {
    client: crate::prompt_manager::prompt_layer::PromptLayerClient,
}

impl PromptLayerRenderer {
    pub fn new(client: crate::prompt_manager::prompt_layer::PromptLayerClient) -> Self {
        Self { client }
    }
}

#[async_trait]
impl PromptRenderer for PromptLayerRenderer {
    async fn render(
        &self,
        template: PromptTemplate,
        variables: &HashMap<String, String>,
        label: Option<&str>,
    ) -> Result<String, crate::prompt_manager::prompt_layer::PromptLayerError> {
        crate::prompt_manager::prompt_layer::get_and_render_prompt(
            &self.client,
            template,
            variables,
            label,
        )
        .await
    }
}

pub async fn choose_the_best_item_for_purpose_from_list<R>(
    labeled_items: &HashMap<String, String>,
    purpose: &str,
    token_counter: &TokenCounter,
    item_type: &str,
    llm: &impl BaseLLM,
    renderer: &R,
    best_item_with_prob_threshold: Option<f64>,
) -> Result<Option<String>, SelectionError>
where
    R: PromptRenderer,
{
    if labeled_items.is_empty() {
        return Err(SelectionError::InvalidInput(
            "labeled_items is empty".into(),
        ));
    }
    if purpose.trim().is_empty() {
        return Err(SelectionError::InvalidInput("purpose is empty".into()));
    }

    for label in labeled_items.keys() {
        if label.len() != 1 {
            return Err(SelectionError::InvalidLabel(label.clone()));
        }
        let ch = label.chars().next().unwrap();
        if !(ch.is_ascii_digit() || (ch.is_ascii_uppercase() && ch.is_ascii_alphabetic())) {
            return Err(SelectionError::InvalidLabel(label.clone()));
        }
    }

    let mut available_labels: Vec<_> = labeled_items.keys().cloned().collect();
    available_labels.sort();
    let labels_str = available_labels.join(", ");
    let labeled_items_list: String = available_labels
        .iter()
        .map(|label| format!("{}: {}", label, labeled_items[label]))
        .collect::<Vec<_>>()
        .join("\n");

    let mut template_vars = HashMap::new();
    template_vars.insert("purpose".to_string(), purpose.to_string());
    template_vars.insert("item_type".to_string(), item_type.to_string());
    template_vars.insert("labels_str".to_string(), labels_str.clone());
    template_vars.insert("labeled_items_list".to_string(), labeled_items_list.clone());

    let system_prompt = renderer
        .render(
            PromptTemplate::SelectionUtilsSystemPrompt,
            &template_vars,
            None,
        )
        .await?;
    let user_prompt = renderer
        .render(
            PromptTemplate::SelectionUtilsUserPrompt,
            &template_vars,
            None,
        )
        .await?;

    let result = llm
        .generate(
            "gpt-4.1",
            &system_prompt,
            &[Message::new(Role::User, user_prompt)],
            token_counter,
            true,
            None,
            Some(1),
        )
        .await?;

    match result {
        GenerateResponse::Logprobs(logprobs) => {
            let mut prob_map: HashMap<String, f64> = HashMap::new();
            for (token, logprob) in logprobs {
                let norm_token = token.trim().to_uppercase();
                let prob = logprob.exp();
                *prob_map.entry(norm_token).or_insert(0.0) += prob;
            }

            if let Some(threshold) = best_item_with_prob_threshold {
                for label in &available_labels {
                    if let Some(prob) = prob_map.get(label)
                        && *prob > threshold
                    {
                        return Ok(Some(label.clone()));
                    }
                }

                for special in ["X", "Y"] {
                    if let Some(prob) = prob_map.get(special)
                        && *prob > threshold
                    {
                        return Ok(None);
                    }
                }

                info!(
                    "No {} exceeded the probability threshold {:.6}",
                    item_type, threshold
                );
                return Ok(None);
            }

            let mut sorted_probs: Vec<_> = prob_map.into_iter().collect();
            sorted_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if let Some((decision, _confidence)) = sorted_probs.first() {
                match decision.as_str() {
                    "X" | "Y" => Ok(None),
                    label if labeled_items.contains_key(label) => Ok(Some(label.to_string())),
                    _ => Err(SelectionError::InvalidLLMResponse),
                }
            } else {
                Err(SelectionError::InvalidLLMResponse)
            }
        }
        GenerateResponse::Text(_) => Err(SelectionError::InvalidLLMResponse),
    }
}
