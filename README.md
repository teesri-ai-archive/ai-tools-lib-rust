# Rust AI Tools Lib (Rust)

`ai-tools-lib-rust` mirrors the Python `ai-tools` package for Rust consumers.
It provides token counting, prompt management (@ PromptLayer), selection helpers,
LLM adapters (OpenAI/Gemini), and gated serialization that respects
`SkipJsonSchema` semantics. This crate is designed for integration in the
`flixie-e2e` workspace and exposes the same validation rules and behaviors as
the existing Python codebase.

## Development

- `cargo build`
- `cargo test`
- `task build` (via shared task definitions)

Refer to the `.github/workflows/ci.yml` file for how CI bootstraps the
workspace and runs the Taskfile.

## Workspace registration

This package is included in the [`flixie-e2e.code-workspace`](../flixie-e2e.code-workspace) so the Rust tooling
can be launched from the monorepo root. After adding new folders, make sure
to keep that file in sync so editors see the project.

## GitHub onboarding

To publish the crate under the `teesri-ai` GitHub organization:

```sh
  gh repo create teesri-ai/ai-tools-lib-rust \
  --description "Rust mirror of the ai-tools utilities" \
  --public \
  --source ./ai-tools-lib-rust \
  --remote origin \
  --push
```

This will create the remote, add it as `origin`, and push the current
worktree. You can then tag & release following the same workflow as other
shared libraries in the monorepo.

## CI credentials

The shared workflows expect a personal access token called `MANISH_GITHUB_PAT`
with at least the following scopes:

- `repo` (full control of private repositories)
- `workflow` (to trigger and manage workflows across repos)
- `read:org` (to list organization resources during checkout)
- `write:packages` (if you need to publish packages from CI)

Create the token under [github.com/settings/tokens](https://github.com/settings/tokens),
then add it to each repository that participates in the Rust CI jobs by visiting
`Settings → Secrets and variables → Actions` and defining `MANISH_GITHUB_PAT`.
