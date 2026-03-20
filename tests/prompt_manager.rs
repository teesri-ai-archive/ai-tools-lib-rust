use ai_tools_lib_rust::{
    prompt_manager::PromptTemplateData,
    prompt_manager::prompt_layer::{
        PromptContent, PromptLayerTemplate, PromptMessage, PromptTemplatePayload, extract_prompt,
        render_f_string, render_jinja, render_prompt,
    },
};

fn sample_template() -> PromptLayerTemplate {
    PromptLayerTemplate {
        prompt_name: "test".into(),
        prompt_template: PromptTemplatePayload {
            messages: vec![PromptMessage {
                input_variables: vec!["name".into(), "age".into()],
                template_format: "f-string".into(),
                content: vec![PromptContent {
                    id: None,
                    annotations: None,
                    text: Some("Hello {name}, you are {age}".into()),
                }],
            }],
        },
    }
}

fn sample_jinja_template() -> PromptLayerTemplate {
    PromptLayerTemplate {
        prompt_name: "jinja".into(),
        prompt_template: PromptTemplatePayload {
            messages: vec![PromptMessage {
                input_variables: vec!["name".into(), "city".into()],
                template_format: "jinja2".into(),
                content: vec![PromptContent {
                    id: None,
                    annotations: None,
                    text: Some("Hello {{ name }}, welcome to {{ city }}!".into()),
                }],
            }],
        },
    }
}

#[test]
fn extract_prompt_success() {
    let template = sample_template();
    let prompt = extract_prompt(&template).expect("extracted");
    assert_eq!(prompt.name, "test");
    assert_eq!(prompt.input_variables, vec!["name", "age"]);
    assert_eq!(prompt.template_format, "f-string");
}

#[test]
fn render_f_string_success() {
    let prompt = PromptTemplateData {
        name: "test".into(),
        input_variables: vec!["name".into()],
        template_format: "f-string".into(),
        template_text: "Hi {name}".into(),
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("name".into(), "Alice".into());
    let result = render_f_string(&prompt, &vars).expect("rendered");
    assert_eq!(result, "Hi Alice");
}

#[test]
fn render_f_string_missing_variable() {
    let prompt = PromptTemplateData {
        name: "test".into(),
        input_variables: vec!["name".into(), "age".into()],
        template_format: "f-string".into(),
        template_text: "Hi {name} - {age}".into(),
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("name".into(), "Alice".into());
    assert!(render_f_string(&prompt, &vars).is_err());
}

#[test]
fn render_jinja_success() {
    let template = sample_jinja_template();
    let prompt = extract_prompt(&template).expect("prompt");
    let mut vars = std::collections::HashMap::new();
    vars.insert("name".into(), "Bob".into());
    vars.insert("city".into(), "NYC".into());
    let rendered = render_jinja(&prompt, &vars).expect("rendered");
    assert_eq!(rendered, "Hello Bob, welcome to NYC!");
}

#[test]
fn render_prompt_invalid_format() {
    let prompt = PromptTemplateData {
        name: "test".into(),
        input_variables: vec!["name".into()],
        template_format: "invalid".into(),
        template_text: "".into(),
    };
    let vars = std::collections::HashMap::new();
    assert!(render_prompt(&prompt, &vars).is_err());
}
