use ai_tools_lib_rust::utils::pydantic_helper::{
    SkipJsonSchema, build_exclude_dict, gated_serializer, is_field_skipped,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
struct SecretConfig {
    api_key: String,
    timeout: u32,
}

impl SkipJsonSchema for SecretConfig {
    fn skip_json_schema_fields(&self) -> Option<Value> {
        let mut map = serde_json::Map::new();
        map.insert("api_key".to_string(), Value::Bool(true));
        Some(Value::Object(map))
    }
}

#[derive(Serialize)]
struct User {
    username: String,
    password_hash: String,
}

impl SkipJsonSchema for User {
    fn skip_json_schema_fields(&self) -> Option<Value> {
        let mut map = serde_json::Map::new();
        map.insert("password_hash".to_string(), Value::Bool(true));
        Some(Value::Object(map))
    }
}

#[derive(Serialize)]
struct Group {
    name: String,
    admin: User,
    members: Vec<User>,
    config: SecretConfig,
}

impl SkipJsonSchema for Group {
    fn skip_json_schema_fields(&self) -> Option<Value> {
        let mut map = serde_json::Map::new();
        if let Some(admin_map) = self.admin.skip_json_schema_fields() {
            map.insert("admin".to_string(), admin_map);
        }
        if let Some(members_map) = self.members.skip_json_schema_fields() {
            map.insert("members".to_string(), members_map);
        }
        if let Some(config_map) = self.config.skip_json_schema_fields() {
            map.insert("config".to_string(), config_map);
        }
        if map.is_empty() {
            None
        } else {
            Some(Value::Object(map))
        }
    }
}

#[test]
fn gated_serializer_respects_skip_fields() {
    let admin = User {
        username: "admin".into(),
        password_hash: "secret".into(),
    };
    let member = User {
        username: "guest".into(),
        password_hash: "guest".into(),
    };
    let config = SecretConfig {
        api_key: "sk-123".into(),
        timeout: 30,
    };
    let group = Group {
        name: "Team".into(),
        admin,
        members: vec![member],
        config,
    };

    let ungated = gated_serializer(&group, false).expect("serialize");
    assert!(ungated.get("admin").is_some());
    assert!(ungated.get("config").is_some());

    let gated = gated_serializer(&group, true).expect("gated");
    let admin = gated.get("admin").and_then(|v| v.as_object()).unwrap();
    assert!(!admin.contains_key("password_hash"));
    assert!(
        gated
            .get("config")
            .unwrap()
            .as_object()
            .unwrap()
            .contains_key("timeout")
    );
}

#[test]
fn build_exclude_dict_detects_nested_skips() {
    let user = User {
        username: "user".into(),
        password_hash: "hash".into(),
    };
    let exclude = build_exclude_dict(&user).expect("exclude");
    assert!(is_field_skipped(Some(&exclude), "password_hash"));
}
