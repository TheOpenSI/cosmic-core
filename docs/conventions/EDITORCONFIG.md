# EditorConfig conventions

This repository uses [EditorConfig](https://editorconfig.org) so every contributor's editor applies
the same indentation, line endings, character set and whitespace rules. The rules live in the root
`.editorconfig` file and are enforced in CI by `.github/scripts/check_editorconfig.py`.

## Rules

| Property | Value | Applies to |
| --- | --- | --- |
| `charset` | `utf-8` | all files |
| `end_of_line` | `lf` | all files |
| `insert_final_newline` | `false` | all files |
| `trim_trailing_whitespace` | `true` | all files except Markdown and `.txt` |
| `indent_style` | `space`, width 4 | Python, shell and other source files |
| `indent_style` | `space`, width 2 | JSON, YAML, TOML, Markdown |
| `spelling_language` | `en-AU` | all files |

Markdown keeps trailing whitespace because a double space at the end of a line is a hard line
break. `.txt` prompt files keep their literal layout, so indentation and trailing whitespace are
not enforced there.

## Editor setup

Install the EditorConfig plugin for your editor (many editors ship it), open the repository root as
the workspace, and the root `.editorconfig` is picked up automatically. Some editors need
EditorConfig enabled in settings.

## Checking your changes

If your editor has an EditorConfig plugin, it applies these rules when you save; that is usually
enough. If it has no plugin, or you want to check from a terminal:

```bash
docker run --rm -v "$PWD:/check" ghcr.io/editorconfig-checker/editorconfig-checker:4.0.2
```

Prebuilt binaries are also on the [editorconfig-checker releases page](https://github.com/editorconfig-checker/editorconfig-checker/releases).

## CI check

Every push and pull request runs [editorconfig-checker](https://github.com/editorconfig-checker/editorconfig-checker)
through `.github/workflows/editorconfig.yml`. It reads the root `.editorconfig` and
`.editorconfig-checker.json`, which lists the paths the checker skips, such as `LICENSE`,
`CONTRIBUTING` and generated or vendored files. It fails with the file and line it found.

The indentation width (`indent_size`) is applied by your editor; CI checks the
indentation style and regular content rules. Git line endings are also normalised
by `.gitattributes` (`* text=auto eol=lf`).

## References

- [EditorConfig specification](https://spec.editorconfig.org/)
- [Projects using EditorConfig](https://github.com/editorconfig/editorconfig/wiki/Projects-Using-EditorConfig)
- [Live CoSMIC docs page](https://github.com/TheOpenSI/cosmic-docs/blob/main/docs/06-development/editorconfig.md)