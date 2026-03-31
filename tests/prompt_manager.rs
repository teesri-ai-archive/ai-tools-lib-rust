use ai_tools_lib_rust::prompt_manager::PromptTemplate;
use ai_tools_lib_rust::prompt_manager::prompt_template::{
    PromptTemplateData, PromptTemplateError, get_prompt_template, render_prompt,
};
use serde_json::Value;

#[test]
fn render_prompt_success() {
    let prompt = PromptTemplateData {
        name: "greet".into(),
        template_text: "Hello {{ name }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "name".into(),
                var_type: "str".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("name".into(), Value::String("Alice".into()));
    let rendered = render_prompt(&prompt, &vars).expect("should render");
    assert_eq!(rendered, "Hello Alice");
}

#[test]
fn render_prompt_missing_variable() {
    let prompt = PromptTemplateData {
        name: "greet".into(),
        template_text: "Hello {{ name }} {{ surname }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "name".into(),
                var_type: "str".into(),
                required: true,
            },
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "surname".into(),
                var_type: "str".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("name".into(), Value::String("Alice".into()));
    assert!(render_prompt(&prompt, &vars).is_err());
}

#[test]
fn render_prompt_extra_variable() {
    let prompt = PromptTemplateData {
        name: "greet".into(),
        template_text: "Hello {{ name }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "name".into(),
                var_type: "str".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("name".into(), Value::String("Alice".into()));
    vars.insert("unused".into(), Value::String("value".into()));
    assert!(render_prompt(&prompt, &vars).is_err());
}

#[test]
fn render_prompt_rejects_string_for_int_type() {
    let prompt = PromptTemplateData {
        name: "n".into(),
        template_text: "n={{ n }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "n".into(),
                var_type: "int".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("n".into(), Value::String("42".into()));
    assert!(render_prompt(&prompt, &vars).is_err());
}

#[test]
fn render_prompt_accepts_json_integer() {
    let prompt = PromptTemplateData {
        name: "n".into(),
        template_text: "n={{ n }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "n".into(),
                var_type: "int".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("n".into(), serde_json::json!(7));
    let rendered = render_prompt(&prompt, &vars).expect("render");
    assert!(rendered.contains('7'));
}

#[test]
fn render_prompt_rejects_bool_for_int_type() {
    let prompt = PromptTemplateData {
        name: "n".into(),
        template_text: "n={{ n }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "n".into(),
                var_type: "int".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("n".into(), Value::Bool(true));
    let err = render_prompt(&prompt, &vars).expect_err("bool is not int");
    assert!(matches!(
        err,
        PromptTemplateError::VariableValueType {
            ref name,
            ref expected_type,
            ..
        } if name == "n" && expected_type == "int"
    ));
}

#[test]
fn render_prompt_rejects_float_json_for_int_type() {
    let prompt = PromptTemplateData {
        name: "n".into(),
        template_text: "n={{ n }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "n".into(),
                var_type: "int".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("n".into(), serde_json::json!(3.5));
    assert!(matches!(
        render_prompt(&prompt, &vars),
        Err(PromptTemplateError::VariableValueType { .. })
    ));
}

#[test]
fn render_prompt_rejects_int_for_bool_type() {
    let prompt = PromptTemplateData {
        name: "b".into(),
        template_text: "b={{ b }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "b".into(),
                var_type: "bool".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("b".into(), serde_json::json!(1));
    assert!(matches!(
        render_prompt(&prompt, &vars),
        Err(PromptTemplateError::VariableValueType { .. })
    ));
}

#[test]
fn render_prompt_list_int_rejects_string_element() {
    let prompt = PromptTemplateData {
        name: "xs".into(),
        template_text: "{% for x in xs %}{{ x }}{% endfor %}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "xs".into(),
                var_type: "list[int]".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("xs".into(), serde_json::json!([1, "2"]));
    let err = render_prompt(&prompt, &vars).expect_err("element type");
    assert!(matches!(err, PromptTemplateError::VariableValueType { .. }));
}

#[test]
fn render_prompt_dict_str_str_accepts_string_values() {
    let prompt = PromptTemplateData {
        name: "d".into(),
        template_text: "{{ data.k }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "data".into(),
                var_type: "dict[str, str]".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("data".into(), serde_json::json!({"k":"v"}));
    let rendered = render_prompt(&prompt, &vars).expect("render");
    assert_eq!(rendered, "v");
}

#[test]
fn render_prompt_dict_str_str_rejects_non_string_value() {
    let prompt = PromptTemplateData {
        name: "d".into(),
        template_text: "{{ data }}".into(),
        variables: vec![
            ai_tools_lib_rust::prompt_manager::prompt_template::PromptVariable {
                name: "data".into(),
                var_type: "dict[str, str]".into(),
                required: true,
            },
        ],
    };
    let mut vars = std::collections::HashMap::new();
    vars.insert("data".into(), serde_json::json!({"k": 1}));
    let err = render_prompt(&prompt, &vars).expect_err("value type");
    assert!(matches!(
        err,
        PromptTemplateError::VariableValueType {
            ref name,
            ref expected_type,
            ..
        } if name == "data" && expected_type == "dict[str, str]"
    ));
}

#[test]
fn get_prompt_template_returns_not_found_when_prompt_dir_missing_from_embed() {
    // Embedded ``flixie-prompts`` in this workspace may only ship a subset of enum paths.
    assert!(matches!(
        get_prompt_template(PromptTemplate::VideoAnalysisPrompt),
        Err(PromptTemplateError::NotFound(_))
    ));
}
