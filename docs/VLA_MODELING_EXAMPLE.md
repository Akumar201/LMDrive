# Understanding LMDrive as a VLA: A Worked Example

This doc is for people new to **Vision-Language-Action (VLA)** models. It explains what the "modeling" side does and walks through one forward pass with concrete shapes and a minimal example.

![VLA modeling pipeline](../assets/vla_modeling_pipeline.png)

---

## 1. What is a VLA?

A **VLA** is a model that:

- Takes **vision** (images, sometimes other sensors like LiDAR),
- Takes **language** (e.g. an instruction: *"Turn left at the next intersection"*),
- Produces **actions** (e.g. steer, throttle, brake) instead of text.

So: **see + read → act**, in one model. LMDrive is a VLA for driving: it watches the road and follows language instructions to output vehicle control.

---

## 2. The three modeling stages (big picture)

Conceptually, the model does:

1. **Encode**  
   Turn raw sensors + instruction into a single “understanding” representation.

2. **Reason**  
   Run that representation through a large language model (LLaMA). We don’t use its text output; we use its **internal representation** (hidden states).

3. **Decode to action**  
   Small “heads” on top of those hidden states predict **waypoints** (where to go) and **end-of-maneuver** (when the instruction is done), then a PID controller turns that into steer/throttle/brake.

The rest of this doc makes that concrete with shapes and one minimal example.

---

## 3. Minimal setup for the example

To keep numbers small, imagine:

- **Batch size** `B = 1`
- **Time steps (frames)** `T = 2`
- **Instruction:** *"Turn left at the next intersection."*
- **Visual encoder** output feature dim: `256`
- **Q-Former** query length: `4` tokens per frame
- **LLM** (LLaMA-7B) hidden size: `4096`

All shapes below follow this setup.

---

## 4. Every step with one running example

Below we trace **one single forward pass** from the simulator to the control output. At every step we show: **what goes in**, **what happens**, and **what comes out**, with concrete numbers for our example (1 batch, 2 frames, instruction *"Turn left at the next intersection."*).

---

### Step 0: Raw inputs (what the world gives the model)

**What goes in (from CARLA):**

| Input | Example in our run |
|--------|---------------------|
| **rgb_front** | Tensor shape `(1, 2, 3, 900, 1200)` — 1 batch, 2 time steps, 3 channels, height 900, width 1200. |
| **rgb_left, rgb_right, rgb_rear** | Similar: `(1, 2, 3, 300, 400)` each. |
| **lidar** | `(1, 2, 40000, 4)` — 40k points per frame, 4 values (x, y, z, intensity). |
| **velocity** | `(1, 2, 1)` — e.g. `[[5.2], [5.4]]` (m/s) for the two frames. |
| **Instruction (string)** | `"Turn left at the next intersection."` |
| **target_point** | `(1, 2)` — e.g. `[[10.0, -3.5]]` (next goal in ego frame). |

**What happens:** Nothing yet. This is the raw tick from the simulator.

**What comes out:** The same tensors and string, passed to the next step. So “modeling” starts from these **sensors + one language string**.

---

### Step 1: Visual encoder (MemFuser)

**What goes in:** The dict above. Inside the model, batch and time are often flattened so the encoder sees `(2, 3, H, W)` for RGB and `(2, 40000, 4)` for LiDAR, etc.

**What happens:** MemFuser runs:
- CNN backbones on each view and on LiDAR.
- Velocity is embedded and fused.
- Features are combined into a single grid of “patch” vectors per frame.

**What comes out (example):**

- **image_embeds:** `(2, 2500, 256)`  
  - 2 = number of frames (we flattened batch×time).  
  - 2500 = number of spatial “patches” (e.g. 50×50).  
  - 256 = feature dimension.  

So after Step 1 we have **2500 vectors of size 256 per frame**, summarizing “what the car sees” (no language yet).

---

### Step 2: LayerNorm (ln_vision)

**What goes in:** `image_embeds` shape `(2, 2500, 256)`.

**What happens:** Layer normalization on the last dimension (256): subtract mean, divide by std, scale and shift with learnable parameters. Stabilizes training.

**What comes out:** Same shape `(2, 2500, 256)`; only the values are normalized.

---

### Step 3: Q-Former (vision + language fusion)

**What goes in:**

- **image_embeds:** `(2, 2500, 256)` (normalized visual features).
- **Instruction:** For each of the 2 frames we use the same text: *"Turn left at the next intersection."* Tokenized → e.g. 10 token IDs, repeated for 2 frames → `(2, 10)`.
- **Query tokens:** 4 learnable vectors per frame, shape `(2, 4, 768)`.

**What happens:** The Q-Former (BERT-style cross-attention):
- **Queries:** the 4 vectors (what we want to “ask”).
- **Keys/Values:** the 2500 image patch vectors plus the 10 text token embeddings.
- It outputs 4 vectors per frame that “answer” the queries using both image and text.

So each of the 4 output vectors per frame encodes “instruction + this frame’s scene”.

**What comes out (example):**

- **query_output.last_hidden_state:** `(2, 4, 768)`  
  - 2 frames, 4 “summary” tokens per frame, dimension 768.  
So we now have **4 tokens per frame** that mix vision and language.

---

### Step 4: LLM projection (llm_proj)

**What goes in:** `(2, 4, 768)` — the Q-Former output.

**What happens:** One linear layer: `y = x @ W + b`, with `W` of shape `(768, 4096)`. Each 768-d vector is mapped to 4096-d (LLaMA’s embedding size).

**What comes out (example):**

- **image_tokens for LLM:** `(2, 4, 4096)`  
  - Same 2 frames and 4 tokens per frame, but now in the **same space as LLaMA’s word embeddings**.  
So we have **4 “image tokens” per frame** that the LLM will process like tokens.

---

### Step 5: Text embeddings (instruction only)

**What goes in:** The string *"Turn left at the next intersection."*

**What happens:**
1. Tokenize with the LLaMA tokenizer → e.g. 10 token IDs: `[1, 523, 291,  ...]` (example).
2. Look up each ID in the LLM’s embedding table → 10 vectors of size 4096.

**What comes out (example):**

- **text_embeds:** `(1, 10, 4096)`  
  - 1 batch, 10 tokens, 4096-d each.  
So the instruction is now **10 vectors** in the same 4096-d space as the image tokens.

---

### Step 6: Concatenate into one sequence (concat_text_image_input)

**What goes in:**

- **text_embeds:** `(1, 10, 4096)` — the 10 instruction tokens.
- **image_tokens frame 1:** `(1, 4, 4096)`.
- **image_tokens frame 2:** `(1, 4, 4096)`.

**What happens:** We concatenate along the **sequence dimension** to build one long sequence for the LLM:

- Position 0–9: instruction tokens.  
- Position 10–13: frame 1’s 4 image tokens.  
- Position 14–17: frame 2’s 4 image tokens.

We also record **wp_target_index**: “at which position do we want to read the LLM’s output for waypoints?” We choose the **last token of each frame’s image block**: position 13 for frame 1 and position 17 for frame 2. So `wp_target_index = [(0, 12), (0, 16)]` (0-based: 12 and 16).

**What comes out (example):**

- **llm_inputs:** `(1, 18, 4096)` — one sequence of 18 tokens.  
- **wp_target_index:** `[(0, 12), (0, 16)]` — we will use hidden states at positions 12 and 16 for the driving heads.

---

### Step 7: LLaMA forward (custom: return hidden states)

**What goes in:** `llm_inputs` `(1, 18, 4096)` and an attention mask (which positions to attend to).

**What happens:** The LLaMA transformer runs:
- 32 layers of self-attention + MLP.
- Each layer updates the hidden state at every position using the full sequence.
- At the end we have a 4096-d vector at each of the 18 positions.  
The **custom** part: we **return these vectors** and do **not** apply the language-model head (no logits, no next-token prediction).

**What comes out (example):**

- **hidden_states:** `(1, 18, 4096)`  
  - One 4096-d vector per position. Position 12 = “LLM’s understanding after seeing instruction + frame 1”; position 16 = “after instruction + frame 1 + frame 2”.

---

### Step 8: Driving heads (waypoints + end)

**What goes in:** We take **only** the hidden states at the frame positions: `hidden_states[:, [12, 16], :]` → `(1, 2, 4096)`.

**What happens:**
- **Waypoints head:** Two linear layers (e.g. 4096→4096→10). Input `(1, 2, 4096)` → output `(1, 2, 10)`. We interpret the 10 numbers as 5 waypoints (x, y) per frame.
- **End head:** Two linear layers (e.g. 4096→4096→2). Input `(1, 2, 4096)` → output `(1, 2, 2)` logits. Softmax → probability “maneuver ended” per frame.

For **control** we use only the **last frame** (index 1): its 5 waypoints and its end probability.

**What comes out (example):**

- **waypoints (last frame):** 5 points in (x, y), e.g. `[[2.1, 0.0], [4.3, -0.5], [6.2, -1.1], [8.0, -1.8], [10.0, -2.5]]` (meters ahead and left).
- **end_prob (last frame):** e.g. `0.15` → 15% chance the instruction is done.

---

### Step 9: From waypoints to control (PID)

**What goes in:** The 5 waypoints (x, y) and current velocity (e.g. 5.4 m/s).

**What happens:** A PID controller (not learned):
- Computes desired steering to follow the waypoints.
- Computes throttle/brake to match a target speed.
- If **end_prob** is high (e.g. > 0.75), the agent may start the next instruction or slow down.

**What comes out (example):**

- **steer:** e.g. `0.12` (slight left).
- **throttle:** e.g. `0.4`.
- **brake:** e.g. `0.0`.

These three numbers are sent to CARLA as the **action** for this tick. The loop then repeats: new sensors → same steps → new steer/throttle/brake.

---

## 5. Step-by-step reference (shapes only)

Compact reference for the same steps (see section 4 for the running example).

### Step 1: Visual encoder (MemFuser)

**What it does:** Fuses multi-view RGB + LiDAR + velocity into one visual representation.

**Input:** A dict of tensors (rgb_front, rgb_left, rgb_right, rgb_rear, lidar, velocity, …), often flattened over batch and time for the backbone.

**Output:** One feature tensor per “frame” (per time step).

- After the encoder and flattening over batch×time, you get something like:
  - `image_embeds`: `(B*T, N_patches, D_vis)`  
  Example: `(2, 2500, 256)` for 2 frames and 256-d features.

So **sensors → one 256-d vector per patch per frame**. No language yet.

---

### Step 2: LayerNorm (ln_vision)

**What it does:** Normalizes the visual features (standard in transformers).

- Input/output shape unchanged: `(2, 2500, 256)`.

---

### Step 3: Q-Former (vision–language fusion)

**What it does:** Asks: *“Given this image and this instruction, what are the 4 most relevant ‘query’ vectors?”* So the **same instruction** conditions the **image** of each frame.

- **Query tokens:** `(2, 4, 768)` — 4 learnable queries per frame, dim 768 (Q-Former hidden size).
- **Instruction:** Tokenized, e.g. 10 tokens → `(2, 10)` (we repeat for each frame).
- **Image:** `image_embeds` `(2, 2500, 256)` as encoder hidden states.

The Q-Former (a BERT-like cross-attention block) does:

- **Query** = 4 tokens (what we want to “ask”),
- **Key/Value** = image patches + instruction tokens,

and outputs 4 vectors per frame that mix **vision + language**.

**Output:**

- `query_output.last_hidden_state`: `(2, 4, 768)`  
  So we get **4 tokens per frame**, each 768-d, that encode “instruction + this frame’s scene”.

---

### Step 4: LLM projection (llm_proj)

**What it does:** Maps Q-Former output (768-d) into the **LLM’s embedding space** (4096-d for LLaMA-7B).

- **Input:** `(2, 4, 768)`
- **Operation:** Linear: `768 → 4096`
- **Output:** `(2, 4, 4096)` — “image + instruction” in LLaMA’s world.

We will treat these as **4 “image tokens” per frame** for the LLM.

---

### Step 5: Text embeddings (instruction only)

**What it does:** Turn the instruction into the same 4096-d embedding space.

- **Input:** String *"Turn left at the next intersection."*
- **Tokenization:** e.g. 10 tokens → ids `(1, 10)`.
- **Embedding:** `llm_model.get_input_embeddings()(ids)` → `(1, 10, 4096)`.

So we have **10 text tokens**, each 4096-d.

---

### Step 6: Concatenate into one sequence (concat_text_image_input)

**What it does:** Build the **single sequence** the LLM will process:  
`[instruction tokens] [frame1 image tokens] [frame2 image tokens]`.

- **Text (before images):** 10 tokens → shape `(1, 10, 4096)`.
- **Frame 1 image:** 4 tokens → `(1, 4, 4096)`.
- **Frame 2 image:** 4 tokens → `(1, 4, 4096)`.

**Concatenate along sequence length:**

- `llm_inputs`: `(1, 10 + 4 + 4, 4096)` = `(1, 18, 4096)`.

So the LLM sees **one sequence of 18 tokens**: first 10 = instruction, next 4 = frame 1, last 4 = frame 2.

We also compute **wp_target_index**: the **positions** where we want to read the LLM’s representation to predict waypoints. Typically: “last token of each frame’s image block”:

- Frame 1 image ends at index 13 (0-based: 9 + 4 - 1 = 12, so position 12).
- Frame 2 image ends at index 17 (12 + 4 = 16, last is 16).

So we might have `wp_target_index = [(0, 12), (0, 16)]` (batch_idx, token_idx). Those are the two positions we will use for the driving heads.

---

### Step 7: LLaMA forward (custom: return hidden states)

**What it does:** Run the full LLaMA stack (many transformer layers) on the 18-token sequence. **We do not use the logits (next-token prediction).** We only use the **last-layer hidden states**.

- **Input:** `llm_inputs` `(1, 18, 4096)`, attention_mask, etc.
- **Inside LLaMA:** 32 layers of self-attention + MLP; each layer updates the hidden state.
- **Output (custom return):** `hidden_states`: `(1, 18, 4096)`.

So for **each** of the 18 positions we have one 4096-d vector = “what the model has understood at that token”.

---

### Step 8: Driving heads (waypoints + end)

**What it does:** Use the hidden state **only at the frame positions** we care about (wp_target_index), and map them to waypoints and end probability.

- **Select:** `hidden_states[:, [12, 16], :]` → `(1, 2, 4096)` (2 positions, one per frame).
- **Waypoints head:**  
  - Input: `(1, 2, 4096)`  
  - Output: e.g. `(1, 2, 10)` → reshape to 5 waypoints (x,y) per frame: `(1, 2, 5, 2)`.
- **End head:**  
  - Input: `(1, 2, 4096)`  
  - Output: `(1, 2, 2)` logits → softmax → P(end) per frame.

For **inference** we usually take the **last frame’s** waypoints and end prob:

- Waypoints: `(1, 5, 2)` — 5 (x,y) points.
- End prob: scalar (e.g. 0.2 = 20% “maneuver done”).

---

### Step 9: From waypoints to control (PID)

**What it does:** This is **not** part of the neural net. A PID controller takes:

- The 5 waypoints (where to go),
- Current velocity,

and outputs **steer, throttle, brake** (continuous values). Those go to the simulator (CARLA) as the **action** of the VLA.

So the **modeling** part ends at the waypoints and end probability; the rest is classical control.

---

## 6. One diagram with shapes (minimal example)

```text
Instruction: "Turn left at the next intersection."
Tokenize → (1, 10) ids → embed → (1, 10, 4096)

Sensors (2 frames) → Visual encoder → (2, 2500, 256)
                    → ln_vision     → (2, 2500, 256)
                    → Q-Former      → (2, 4, 768)
                    → llm_proj      → (2, 4, 4096)

Concat: [ (1,10,4096) | (1,4,4096) | (1,4,4096) ] → llm_inputs (1, 18, 4096)
                                                     wp_target_index = [12, 16]

LLaMA(llm_inputs) → hidden_states (1, 18, 4096)

hidden_states[:, [12,16], :] → (1, 2, 4096)
    → waypoints_predictor → (1, 2, 10) → use last frame → 5 waypoints (x,y)
    → end_predictor       → (1, 2, 2)  → use last frame → P(end)

Waypoints + velocity → PID → steer, throttle, brake
```

---

## 7. Pseudocode (one forward)

```python
# 0. Raw
rgb, lidar, velocity = sensors()       # e.g. (1,2,...), (1,2,N,4), (1,2,1)
instruction = "Turn left at the next intersection."

# 1. Vision
image_embeds = visual_encoder(rgb, lidar, velocity)  # (2, 2500, 256)
image_embeds = ln_vision(image_embeds)

# 2. Vision + language (Q-Former)
query_tokens = expand(learned_queries, 2)            # (2, 4, 768)
text_ids = tokenize(instruction)                     # (2, 10) for 2 frames
query_out = Qformer(queries=query_tokens, encoder_hidden_states=image_embeds, text_ids=text_ids)
image_tokens = llm_proj(query_out.last_hidden_state) # (2, 4, 4096)

# 3. Text in LLM space
text_embeds = llm_embed(tokenize(instruction))       # (1, 10, 4096)

# 4. One sequence
llm_inputs = concat([text_embeds, image_tokens[0], image_tokens[1]], dim=1)  # (1, 18, 4096)
wp_target_index = [(0, 12), (0, 16)]  # last token of frame1 and frame2

# 5. LLM (custom: return hidden states only)
hidden_states = llm_model(inputs_embeds=llm_inputs)  # (1, 18, 4096)

# 6. Driving heads (at frame positions only)
at_positions = hidden_states[:, [12, 16], :]         # (1, 2, 4096)
waypoints = waypoints_predictor(at_positions)        # (1, 2, 10) → 5 (x,y) per frame
end_logits = end_predictor(at_positions)             # (1, 2, 2)

# 7. Use last frame for control
waypoints_last = waypoints[0, -1].view(5, 2)         # (5, 2)
end_prob = softmax(end_logits[0, -1])[1]
steer, throttle, brake = PID(waypoints_last, velocity)
```

---

## 8. Takeaway for “modeling”

- **VLA:** vision + language in → **actions** out (here: waypoints + end prob, then PID → control).
- **Modeling** here = everything from sensors + text up to (and including) waypoints and end probability.
- The **LLM is used as a feature backbone**: we only use its **hidden states** at a few positions (one per frame), not its text output. Those hidden states are then mapped to **driving outputs** by small heads.
- The **same** instruction conditions every frame (via Q-Former), and the **sequence** [instruction, frame1, frame2, …] lets the LLM “see” the evolution of the scene over time and output a representation we decode into actions.

If you open the code, you can match these steps to `drive.py` (forward, concat_text_image_input, waypoints_predictor, end_predictor) and the agent’s `run_step` that feeds sensors and instruction into the model and then PID.
