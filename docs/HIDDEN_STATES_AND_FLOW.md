# Hidden States and Data Flow in LMDrive

## What are "hidden states"?

In a transformer-based LLM (like LLaMA):

- **Input:** a sequence of token embeddings, shape `(batch, seq_len, hidden_size)` (e.g. 4096-dim for LLaMA-7B).
- **Each layer** takes the previous representation and outputs a new representation for every token position. That output is called the **hidden state** at that layer: a continuous vector per position that encodes “what the model has understood so far” at that position.
- **Last layer’s output** = the final hidden states: shape `(batch, seq_len, hidden_size)`. This is the internal representation *before* the vocabulary (language) head.

So:

- **Hidden states** = internal, continuous representation (e.g. 4096-dim per token). Used for reasoning, retrieval, or downstream heads.
- **Logits** = `lm_head(hidden_states)` = scores over the vocabulary for “next token”. Used only for text generation/loss.

LMDrive uses **hidden states** (internal representation), not logits (next-token prediction).

---

## Why the custom `LlamaForCausalLM` returns hidden states

In `modeling_llama.py` (lines 1045–1065):

```text
outputs = self.model(...)           # LLaMA transformer stack
hidden_states = outputs[0]          # (batch, seq_len, hidden_size)
logits = self.lm_head(hidden_states)
# ... loss computation for training ...
return hidden_states               # ← early return: drive path
# (code below is dead: return_dict, CausalLMOutputWithPast, etc.)
```

So the custom forward **always returns** the last-layer hidden states and never returns logits or `CausalLMOutputWithPast`. The drive code then uses these hidden states as the representation to predict waypoints and end-of-maneuver.

If you replaced this with the standard `transformers.LlamaForCausalLM`, its `forward()` returns `(loss,) + (logits,) + ...` or `CausalLMOutputWithPast` (logits, etc.), **not** the raw hidden states in the same way. The driving pipeline expects `hidden_states` and feeds them into `waypoints_predictor` and `end_predictor`, so swapping to the standard class would break the pipeline without changing how its output is used.

---

## End-to-end data flow (map)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. SENSORS (CARLA)                                                          │
│    RGB (front/left/right/rear), LiDAR, velocity, target_point               │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. VISUAL ENCODER (MemFuser / timm)                                         │
│    input_data → visual_encoder(input_data) → image features                 │
│    Shape: (batch, time, num_patches, feature_dim)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. LN_VISION + Q-FORMER                                                      │
│    image_embeds = ln_vision(image_embeds)                                    │
│    Q-Former: query_tokens + instruction text cross-attend to image_embeds   │
│    query_output = Qformer.bert(..., encoder_hidden_states=image_embeds)       │
│    → "query" tokens that mix instruction + visual info                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. LLM PROJECTION                                                            │
│    image_embeds = llm_proj(query_output.last_hidden_state)                  │
│    Puts visual+instruction representation into LLM embedding space           │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
┌──────────────────────────────┐     ┌──────────────────────────────────────┐
│ 5a. TEXT EMBEDDINGS          │     │ 5b. CONCAT TEXT + IMAGE               │
│  instruction tokenized       │     │ concat_text_image_input():            │
│  → get_input_embeddings()     │     │   [text_before, image_embeds,        │
│  → input_embeds (text only)   │     │    text_after] → llm_inputs           │
└──────────────────────────────┘     │ One long sequence per sample.          │
                                    │ wp_target_index = positions of last   │
                                    │ token of each *frame* (for waypoints). │
                                    └──────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 6. LLM (custom LlamaForCausalLM)                                             │
│    llm_model(inputs_embeds=llm_inputs, attention_mask=..., return_dict=False)│
│    → runs LLaMA transformer stack (many layers)                             │
│    → returns last-layer HIDDEN STATES (batch, seq_len, hidden_size)          │
│    Does NOT return logits; the lm_head and CausalLMOutput are unused.       │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 7. DRIVING HEADS (use hidden states at frame positions only)                │
│    wp_target_index selects positions: last token of each driving frame     │
│    hidden_states[wp_target_index] → waypoints_predictor → (x,y) waypoints  │
│    hidden_states[wp_target_index] → end_predictor → end-of-maneuver prob     │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 8. CONTROL                                                                  │
│    waypoints + end_prob → PID controller → steer, throttle, brake          │
│    → CARLA VehicleControl                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary table

| Stage              | Output / role |
|--------------------|----------------|
| Visual encoder     | Image (+ LiDAR/velocity) → spatial features. |
| Q-Former           | Fuse instruction text + image into “query” tokens in a fixed length. |
| llm_proj           | Map query tokens into LLM embedding space. |
| concat_text_image  | Build one sequence: [text] [frame1 img] [frame2 img] … [text]. |
| LlamaForCausalLM   | Process sequence; **return last-layer hidden states** (no logits). |
| waypoints_predictor| Hidden states at frame positions → 5 waypoints (x,y). |
| end_predictor      | Hidden states at frame positions → P(end of maneuver). |

So: **hidden states** are the LLM’s internal representation of the whole sequence (instruction + multi-frame visuals). LMDrive uses them at specific positions (one per frame) to predict waypoints and end probability; it does not use the language-model output (logits). That is why the custom `LlamaForCausalLM` must return `hidden_states`, and why swapping it for the standard one without changing the rest of the pipeline would break the model.
