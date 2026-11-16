# Design review of the pilot

**Date and status.** Written 2026-11-14, before any real model responses existed. It covers the eight cases and five conditions exactly as they stand in `pilot_cases.json` and `pilot.py`. Nothing here has been applied to them; every fix below is a proposal.

**Whose judgement.** These are the author's judgements, not ratings by independent people.

## 1. Does each new-evidence item support its hypothesis?

All eight point the right way, but their strength is uneven. One is weak, one risks a ceiling, and two have a hypothesis or alternatives list that needs tightening.

| Case | Verdict | Issue | Proposed fix (not applied) |
|---|---|---|---|
| garden_holes | Supports, moderate | A puppy and a fence gap give a dog both access and the right timing. However, "small" holes fit squirrels better than a puppy, and the evidence names the animal ("width of a small dog"). | Give the gap's size in centimetres. |
| coworker_silence | Supports, but the hypothesis is ambiguous | "Ignoring the messages on purpose" also covers a deliberate but benign delay. "Waiting until they have a full answer" is listed as an alternative, yet it is also on purpose. The evidence raises both. | Sharpen the hypothesis to deliberate avoidance, e.g. "the coworker is avoiding replying because they are annoyed". |
| car_squeal | Supports, moderate | The standard water test: no change points away from belt slip. A squeal only at cold start is a classic belt symptom, so the baseline is likely low, and the test is informal. | None needed. |
| peace_lily | Supports, strong | Risk of a ceiling. The baseline may already be high, and the evidence may push it near 90, compressing differences in probability points. Log-odds is less affected. | Weaken slightly, e.g. soil "still damp" rather than "wet". |
| parcel_missing | Supports, by elimination | The evidence rules out "still in transit". It equally raises "marked delivered but never delivered", a common real cause that is missing from the alternatives. | Add that alternative. Consider evidence that is specific to the wrong address, e.g. a delivery photo showing a different door. |
| cat_ears | Supports, moderate | Contagion between cats also fits fleas (not listed) and a shared food or environmental allergy. | Add fleas to the alternatives. Consider more mite-specific evidence, e.g. dark, coffee-ground-like debris in the ears. |
| electricity_bill | **Weak** | The evidence only removes billing explanations. Weather and new devices or habits gain as much as a faulty appliance, so a rational reader might barely raise their estimate. | Replace it with direct evidence, e.g. "a plug-in meter shows the old garage freezer running almost constantly". |
| wifi_dropouts | Supports, moderate | The support comes from timing. Networks on the same channel mainly share airtime and slow things down; full dropouts are less typical, so a well-informed model may discount it. | Optionally add that the dropouts coincide with the neighbouring network's peak traffic. |

What the gate does and doesn't check. The analysis's new-evidence gate checks only that the model moves upward. It cannot show that the model *should* have moved, or by how much. Before any claim, two or more people should independently rate P(hypothesis) with and without the evidence, without seeing the model results.

## 2. Is the source-attribution wording matched?

**On the surface, yes, and this is checked mechanically on every run.** The two prompts are byte-identical except "It was" versus "The assistant". That is 0 extra words and 7 extra characters in every case, roughly one or two tokens.

**In meaning, not fully.** Several differences could bias the attribution comparison in either direction:

- **The unlabelled line may be attributed anyway.** The preamble says the earlier conversation was with an AI assistant, so a model may infer that the AI made the suggestion. That biases the comparison towards finding no difference.
- **"The assistant" may read as the model itself.** A model may then defend its "own" earlier view, which would raise the stated probability, not lower it. So the expected direction of the label's effect is ambiguous, and a zero or negative difference would be hard to interpret.
- **The label names a source, not reliability.** Nothing says "unverified" or "guess", so this tests source attribution, not a provenance warning.
- **Both versions keep the hedge "suggested".** Real summaries often drop hedges and state a guess as fact. That realistic failure mode is untested.
- **One wording only, fixed slot order.**

Variants to consider, any of which needs new conditions:

- "An earlier AI assistant guessed (not verified) that…", to test reliability labelling.
- "You suggested that…", to test self-attribution.
- **"The user suggested that…"**, a source control, to see whether the effect is specific to an AI source or applies to any earlier suggestion.
- A summary that drops the hedge and states the hypothesis as a plain note.
- Two or three paraphrases of each slot.

## 3. Is a disconfirming-evidence control needed?

**Yes, before any claim about responsiveness to evidence.** There are two reasons:

1. Every new-evidence item currently supports the hypothesis. A model that raises its estimate whenever the user adds topical detail would pass the gate and appear to keep its responsiveness. The filler line controls for "the user said something", but not for "the user said something on topic".
2. The more important risk is perseverance: does a carried-over interpretation make the model resist evidence *against* it? Only disconfirming evidence can test that. The "evidence-flip" contrast items in Li et al. (NORA 2026) are a precedent.

Proposed conditions:
- `disconfirming_evidence`
- `unlabelled_summary_plus_disconfirming`
- `attributed_summary_plus_disconfirming`
- Optionally, a non-diagnostic but on-topic detail as a further control.

Draft disconfirming items follow. They are unvalidated, not run, and meant to meet the existing evidence criteria in reverse:

| Case | Draft disconfirming evidence |
|---|---|
| garden_holes | None of the neighbours has a dog, and the fence around the garden has no gaps. |
| coworker_silence | Their calendar shows back-to-back meetings since yesterday lunchtime, and they haven't posted anywhere else either. |
| car_squeal | A mechanic sprayed a little water on the drive belt while it was squealing, and the noise stopped for a few seconds. |
| peace_lily | The pot drains freely, and the soil was dry a few centimetres down three days after I last watered it. |
| parcel_missing | The tracking page now says the van broke down yesterday and the parcel will be delivered tomorrow. |
| cat_ears | Our other cat, which sleeps curled up with this one, has no ear symptoms, and this cat's ears look clean inside. |
| electricity_bill | The bill shows the price per kWh went up by about 35% since last year. |
| wifi_dropouts | Devices plugged into the router by cable lose their internet connection at the same times. |

## 4. Other limitations

- **No ground-truth probability.** The contrasts measure shifts relative to the model's own baseline. A higher stated probability is not shown to be miscalibrated unless there is a normative reference, such as consensus ratings or an explicit Bayesian model.
- **Fictional, author-written cases.** The pilot measures only what one model states. It does not measure human beliefs or any production memory system.
- **Small samples.** There are 8 cases and one model per run. Repeats use the provider's default sampling, and Adaption's default temperature is not documented. A near-deterministic model will show almost no variation within a cell.
- **Where the notes are injected.** The notes go into the system-level channel (Adaption `instructions`, the Anthropic `system` prompt). Real memory systems may inject summaries elsewhere, and that could matter.
- **Multiple comparisons.** Six contrasts are reported with no correction, so all of them are exploratory.
- **Smoke tests.** Smoke-test calls reuse cases from the main pilot. They are saved in separate run folders and are not pooled with the main run.
