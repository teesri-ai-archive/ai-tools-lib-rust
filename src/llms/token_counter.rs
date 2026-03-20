use parking_lot::Mutex;
use serde::Serialize;
use std::collections::HashMap;
use std::fmt;

/// Token counts mirroring the Python implementation.
#[derive(Clone, Debug, Serialize, Default)]
pub struct TokenCounts {
    pub prompt_tokens: u64,
    pub output_tokens: u64,
    pub reasoning_tokens: u64,
}

impl fmt::Display for TokenCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Prompt: {}, Output: {}, Reasoning: {}",
            self.prompt_tokens, self.output_tokens, self.reasoning_tokens
        )
    }
}

/// Thread-safe counter for interaction totals.
#[derive(Default)]
pub struct TokenCounter {
    counts: Mutex<HashMap<String, TokenCounts>>,
}

impl TokenCounter {
    pub fn new() -> Self {
        Self {
            counts: Mutex::new(HashMap::new()),
        }
    }

    pub fn add_counts(
        &self,
        model_name: &str,
        prompt_tokens: u64,
        output_tokens: u64,
        reasoning_tokens: u64,
    ) {
        let mut guard = self.counts.lock();
        let entry = guard.entry(model_name.to_string()).or_default();
        entry.prompt_tokens += prompt_tokens;
        entry.output_tokens += output_tokens;
        entry.reasoning_tokens += reasoning_tokens;
    }

    pub fn get_counts(&self) -> HashMap<String, TokenCounts> {
        self.counts.lock().clone()
    }
}

impl fmt::Display for TokenCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = self.counts.lock();
        if guard.is_empty() {
            write!(f, "Token Counts: (No tokens recorded)")
        } else {
            writeln!(f, "Token Counts:")?;
            for (model, counts) in guard.iter() {
                writeln!(f, "  {}: {}", model, counts)?;
            }
            Ok(())
        }
    }
}
