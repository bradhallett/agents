"""Enconvo adapter (https://enconvo.ai — macOS AI assistant).

Enconvo installs Claude Code plugins as they are, so this adapter exists for one
reason: model resolution. Enconvo honors an agent's `model:` frontmatter only when
the chat's model provider carries a matching model and otherwise runs the helper
on the chat's model — the source aliases (`opus`, `sonnet`, …) name Claude models,
so a user whose providers exclude Anthropic silently loses every tier hint.

The adapter emits self-contained per-plugin trees at `.enconvo/plugins/<plugin>/`
in the Claude Code plugin format Enconvo reads natively, with `model:` resolved to
concrete model IDs from `MODEL_ALIASES["enconvo"]` (defaults to the z.ai GLM
catalog), plus a generated `.enconvo/marketplace.json` pointing at the trees so
`enconvo plugin install .enconvo --plugin <name>` installs a resolved copy.

Structure per plugin:

    .enconvo/plugins/<plugin>/
      .claude-plugin/plugin.json    mirrored verbatim from the source plugin
      .codex-plugin/plugin.json     mirrored when present — Enconvo reads its
                                    `interface` for the plugin's page
      agents/<agent>.md             source frontmatter with `model:` resolved;
                                    body verbatim (Enconvo reads Claude Code
                                    tool names, so nothing is rewritten)
      skills/<skill>/SKILL.md       verbatim frontmatter + body; support files
                                    (references/, assets/, …) mirrored
      commands/<command>.md         verbatim — `$ARGUMENTS` and `` !`cmd` ``
                                    lines work in Enconvo as in Claude Code
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.adapters.base import (
    AgentSource,
    CommandSource,
    EmitResult,
    HarnessAdapter,
    PluginSource,
    SkillSource,
    h1_from_body,
    yaml_scalar,
)
from tools.adapters.capabilities import resolve_model

# Every character YAML treats as structural inside a flow collection. A plain scalar
# in flow context may not contain any of them, at any position.
_YAML_FLOW_DELIMITERS = ("[", "]", "{", "}", ",")


def _yaml_flow_scalar(value: object) -> str:
    """Render a value as one item of a YAML flow sequence (`[a, b]`).

    Flow sequences use `[`, `]`, `{`, `}` and `,` as structural delimiters, so an
    item containing any of them must be quoted even when `yaml_scalar` wouldn't
    quote it as a bare top-level scalar.
    """
    s = str(value).replace("\n", " ")
    if any(delimiter in s for delimiter in _YAML_FLOW_DELIMITERS):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return yaml_scalar(s)


def _enconvo_frontmatter(fm: dict) -> str:
    lines = ["---"]
    for k, v in fm.items():
        if isinstance(v, list):
            value = ", ".join(_yaml_flow_scalar(x) for x in v)
            lines.append(f"{k}: [{value}]")
        elif isinstance(v, dict):
            # Preserve mapping-valued fields (e.g. `metadata`) as a nested YAML
            # mapping instead of stringifying the Python dict repr.
            lines.append(f"{k}:")
            for subk, subv in v.items():
                lines.append(f"  {subk}: {yaml_scalar(subv)}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'true' if v else 'false'}")
        elif v is None:
            continue
        else:
            lines.append(f"{k}: {yaml_scalar(v)}")
    lines.append("---")
    return "\n".join(lines)


class EnconvoAdapter(HarnessAdapter):
    """Emit self-contained Enconvo plugins with models resolved to the user's providers.

    See the module docstring for why a harness that reads the source format still
    needs an adapter: the `model:` tier hints only fire when the chat's provider
    carries a matching model ID, and the source aliases name Claude models.
    Enconvo namespaces colliding helper names itself (plugin prefix) and offers at
    most 20 helpers per chat, so bare `<plugin>`-local names are safe here.
    """

    harness_id = "enconvo"

    def emit_plugin(self, plugin: PluginSource) -> EmitResult:
        result = EmitResult()
        self._emit_manifests(plugin, result)
        for agent in plugin.agents:
            self._emit_agent(plugin, agent, result)
        for skill in plugin.skills:
            self._emit_skill(plugin, skill, result)
        for command in plugin.commands:
            self._emit_command(plugin, command, result)
        return result

    def emit_global(self, plugins: list[PluginSource]) -> EmitResult:
        """Emit `.enconvo/marketplace.json` pointing at the generated plugin trees."""
        return self._emit_marketplace(plugins)

    # ── Internals ──────────────────────────────────────────────────────────

    def _plugin_root(self, plugin: PluginSource) -> Path:
        return Path(".enconvo") / "plugins" / plugin.name

    def _emit_manifests(self, plugin: PluginSource, result: EmitResult) -> None:
        """Mirror the source manifests verbatim.

        `.claude-plugin/plugin.json` is what makes the folder a Claude Code-format
        plugin to Enconvo; `.codex-plugin/plugin.json` is optional but Enconvo reads
        its `interface` listing (display name, default prompts, links, icon) for the
        plugin's page, and the committed Codex manifest already carries it.
        """
        root = self._plugin_root(plugin)
        source_manifest = plugin.dir / ".claude-plugin" / "plugin.json"
        if source_manifest.is_file():
            result.written.append(
                self.mirror_file(source_manifest, root / ".claude-plugin" / "plugin.json")
            )
        else:
            data: dict = {"name": plugin.name}
            if plugin.description:
                data["description"] = plugin.description
            result.written.append(
                self.write(
                    root / ".claude-plugin" / "plugin.json", json.dumps(data, indent=2) + "\n"
                )
            )
        codex_manifest = plugin.dir / ".codex-plugin" / "plugin.json"
        if codex_manifest.is_file():
            result.written.append(
                self.mirror_file(codex_manifest, root / ".codex-plugin" / "plugin.json")
            )

    def _emit_agent(self, plugin: PluginSource, agent: AgentSource, result: EmitResult) -> None:
        """Emit one agent with `model:` resolved; the body is kept verbatim.

        Tool names stay CamelCase (Enconvo reads the Claude Code vocabulary) and
        `color`/`permissionMode` pass through untouched.
        """
        rel = self._plugin_root(plugin) / "agents" / f"{agent.name}.md"

        model, warning = resolve_model("enconvo", agent.model)
        if warning:
            result.warnings.append(f"agent `{plugin.name}__{agent.name}`: {warning}")

        fm = dict(agent.frontmatter)
        # Normalize `tools:` to a real YAML list — the tolerant source parser keeps
        # Claude's bare `tools: Read, Grep` form as a string. Names stay CamelCase:
        # Enconvo reads the Claude Code vocabulary, so this is an identity map.
        if "tools" in agent.frontmatter:
            fm["tools"] = agent.tools
        if model and model != "inherit":
            fm["model"] = model
        else:
            # `inherit` means the chat's model applies — omit the field entirely
            # rather than emitting the literal string.
            fm.pop("model", None)

        content = _enconvo_frontmatter(fm) + "\n\n" + agent.body.rstrip() + "\n"
        result.written.append(self.write(rel, content))

    def _emit_skill(self, plugin: PluginSource, skill: SkillSource, result: EmitResult) -> None:
        """Mirror one skill verbatim (bare name — the plugin dir already scopes it)."""
        rel_dir = self._plugin_root(plugin) / "skills" / skill.name
        fm = dict(skill.frontmatter)
        fm["name"] = skill.name

        content = _enconvo_frontmatter(fm) + "\n\n" + skill.body.rstrip() + "\n"
        result.written.append(self.write(rel_dir / "SKILL.md", content))

        # Mirror every support file (references/, assets/, scripts/, …) — binary copy
        # so non-text assets don't crash the run. Skip hidden files/dirs.
        for src in sorted(skill.dir.rglob("*")):
            if not src.is_file() or src.name == "SKILL.md":
                continue
            rel = src.relative_to(skill.dir)
            if any(part.startswith(".") for part in rel.parts):
                continue
            result.written.append(self.mirror_file(src, rel_dir / rel))

    def _emit_command(
        self, plugin: PluginSource, command: CommandSource, result: EmitResult
    ) -> None:
        """Mirror one slash command verbatim.

        Enconvo installs Claude Code commands as skills named by their file path and
        runs `!`cmd`` lines and `$ARGUMENTS` itself, so the body needs no rewriting.
        """
        rel = self._plugin_root(plugin) / "commands" / f"{command.name}.md"

        fm = dict(command.frontmatter or {})
        if not fm.get("description"):
            fm["description"] = h1_from_body(command.body) or command.name.replace("-", " ").title()

        content = _enconvo_frontmatter(fm) + "\n\n" + command.body.rstrip() + "\n"
        result.written.append(self.write(rel, content))

    def _emit_marketplace(self, plugins: list[PluginSource]) -> EmitResult:
        """Emit `.enconvo/marketplace.json` pointing at the generated plugin trees.

        `enconvo plugin install .enconvo --plugin <name>` resolves each entry's
        `source` relative to the marketplace file, so adding one marketplace covers
        every generated plugin, and re-running the install after a regenerate
        updates the installed copy.
        """
        result = EmitResult()
        entries = []
        for plugin in plugins:
            entry: dict = {"name": plugin.name, "source": f"./plugins/{plugin.name}"}
            if plugin.description:
                entry["description"] = plugin.description
            if plugin.version and plugin.version != "0.0.0":
                entry["version"] = plugin.version
            if plugin.author.get("name"):
                entry["author"] = plugin.author
            entries.append(entry)
        data = {
            "name": "agents-enconvo",
            "owner": {
                "name": "bradhallett",
                "url": "https://github.com/bradhallett/agents",
            },
            "metadata": {
                "description": ("wshobson/agents plugins with model aliases resolved for Enconvo"),
                "version": "1.0.0",
            },
            "plugins": entries,
        }
        result.written.append(
            self.write(Path(".enconvo") / "marketplace.json", json.dumps(data, indent=2) + "\n")
        )
        return result
