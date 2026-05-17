# CommercialBrainEncoder — Claude Instructions

## Session Persistence Protocol

**At the END of every session** (or when context is getting long), you MUST:

1. **Update `brain_encoder_current_status.md`** in Claude memory with:
   - What was accomplished this session
   - Current phase/task
   - Any blockers or decisions made
   - Next steps for the following session

2. **Append a session log entry** in Obsidian at `AI-Employee-Platform/BrainEncoder/07-session-logs/SESSION_LOG.md`:
   - Dated entry with: what changed, files touched, decisions made
   - Keep entries concise (3-5 bullet points per session)

3. **Update any stale memory files** if architecture decisions changed.

4. **Save to Mem0**: Call `mcp__mem0__add-memory` with session summary (userId: "alexx")

5. **Update Obsidian index**: Call `mcp__obsidian-semantic__vault` action="update" on the session log

**At the START of every session**, you MUST:
1. Read `brain_encoder_current_status.md` from memory
2. Read the latest session log entry
3. Resume without asking for a recap

---

## Project Context

Clean-room commercial fMRI brain encoder (MIT license). Maps video+audio stimuli → brain voxel predictions. Beats TRIBE v2 on Pearson r (target >0.23) + adds reconstruction + context heads.

- Obsidian vault: `AI-Employee-Platform/BrainEncoder/` (canonical — use wiki-links here)
- Architecture decisions: `AI-Employee-Platform/BrainEncoder/02-architecture/ARCHITECTURE_DECISIONS_LOG.md`
- Custom skills: `.claude/commands/brain-encoder/` (8 project-specific skills)

---

## Skill Routing — What To Use When

### CommercialBrainEncoder Development
| Task | Skill | Notes |
|---|---|---|
| Add new fMRI dataset | `/brain-encoder:new-dataset` | License gate first — CC0/CC-BY only |
| Add new model component | `/brain-encoder:new-model-component` | Clean-room IP, no TRIBE imports |
| New training experiment | `/brain-encoder:new-experiment` | WandB config + hypothesis |
| Full quality gate | `/brain-encoder:quality-check` | Run before every commit |
| IP + license audit | `/brain-encoder:security-audit` | Run before every release |
| Add dependency | `/brain-encoder:dep-add` | GPL/NC blocked |
| Write tests | `/brain-encoder:test-write` | pytest, small_config fixtures |
| Benchmark vs TRIBE v2 | `/brain-encoder:benchmark` | Pearson r > 0.23 = success |

### Decision Making
| Task | Skill | How to invoke |
|---|---|---|
| Any significant decision | `/decide` | Auto-scales to decision size |
| Minor (naming, param) | `/decide --size minor` | 2-3 agents |
| Medium (library, loss fn) | `/decide --size medium` | 4-6 agents |
| Major (architecture) | `/decide --size major` | 8-12 agents |

### Project Management
| Task | Skill | How to invoke |
|---|---|---|
| Plan a phase | `/gsd:plan-phase` | Creates executable plan |
| Execute a phase | `/gsd:execute-phase` | Wave-based execution |
| Check progress | `/gsd:progress` | Status + next action |
| Debug | `/gsd:debug` | Scientific method |
| Quick task | `/gsd:quick` | Atomic commits |

### Supplementary (fallback only)
- `fullstack-dev-skills:ml-pipeline` — Complex training pipeline patterns beyond skill scope
- `fullstack-dev-skills:python-pro` — Advanced Python async/typing patterns
- `fullstack-dev-skills:fine-tuning-expert` — LoRA/PEFT deep dives

---

## Decision Protocol (Always Active)

Auto-trigger `/decide` for:
- New loss function or architecture change
- New training library (Lightning vs vanilla PyTorch, etc.)
- New dataset integration approach
- API design choices

Never trigger for:
- Following existing patterns (LoRA, Pearson loss, etc.)
- Adding a dataset that's clearly CC0
- Bug fixes with obvious correct solution

Log decisions to: `AI-Employee-Platform/BrainEncoder/02-architecture/ARCHITECTURE_DECISIONS_LOG.md`

---

## Domain Rules (Always Active)

1. **Clean-Room IP**: NEVER import from TRIBE v2, MindEye, BrainBench, or any CC-BY-NC library. All architecture is original.
2. **Dataset License Gate**: EVERY dataset added to `DATASETS` dict must have `commercial_verified: True` and a CC0 or CC-BY license. No exceptions.
3. **No NC Training Data**: Model weights trained on NC-licensed data inherit restrictions. If in doubt, do NOT add the dataset.
4. **Pearson r is Truth**: Evaluation always uses Pearson r computed on held-out val/test split. Never report train metrics as performance.
5. **nn.GELU Standard**: Use GELU activations throughout — not ReLU. Matches existing architecture convention.
6. **Type Annotations Required**: All `forward()` methods have full type annotations. No bare `torch.Tensor` without shape comment.
7. **License in Requirements**: Every line in `requirements.txt` must have a comment justifying it. No mystery dependencies.
8. **Experiment Hypothesis First**: Never start a training run without a documented hypothesis in the experiment config.

---

## Communication Style

Advanced, no-BS. Lead with the answer. Skip preamble. No generic ML advice.
