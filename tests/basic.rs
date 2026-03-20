use ai_tools_lib_rust::{TokenCounter, math_utils::randomized_round};

#[test]
fn token_counter_tracks_counts() {
    let counter = TokenCounter::new();
    counter.add_counts("test-model", 10, 5, 2);

    let counts = counter.get_counts();
    let model_counts = counts.get("test-model").expect("model missing");
    assert_eq!(model_counts.prompt_tokens, 10);
    assert_eq!(model_counts.output_tokens, 5);
    assert_eq!(model_counts.reasoning_tokens, 2);
}

#[test]
fn randomized_round_returns_expected_values() {
    assert_eq!(randomized_round(3.0), 3);
    for _ in 0..5 {
        let result = randomized_round(3.5);
        assert!(result == 3 || result == 4);
    }
}
