# LMDrive Architecture

High-level architecture of the closed-loop driving model (vision encoder → Q-Former → LLM → driving heads).

![LMDrive architecture diagram](../assets/lmdrive_architecture.png)

---

## Block diagram (Mermaid)

```mermaid
flowchart TB
    subgraph sensors["Sensors (CARLA)"]
        RGB[RGB: front, left, right, rear]
        LIDAR[LiDAR]
        VEL[Velocity]
        TGT[Target point]
    end

    subgraph vision["Vision & language fusion"]
        VE["Visual encoder<br/>(MemFuser / timm)"]
        LN["LayerNorm<br/>(ln_vision)"]
        QF["Q-Former<br/>(cross-attn: query + instruction + image)"]
        PROJ["llm_proj<br/>(→ LLM embedding dim)"]
    end

    subgraph text["Instruction"]
        TOK["Tokenize instruction"]
        EMB["LLM input embeddings"]
    end

    subgraph llm["LLM (custom LLaMA)"]
        CONCAT["Concat: text + image tokens<br/>per frame"]
        LLAMA["LlamaForCausalLM<br/>(returns hidden_states)"]
    end

    subgraph heads["Driving heads"]
        WP["waypoints_predictor"]
        END["end_predictor"]
    end

    subgraph out["Output"]
        PID["PID controller"]
        CTRL["steer, throttle, brake"]
    end

    RGB --> VE
    LIDAR --> VE
    VEL --> VE
    TGT --> VE

    VE --> LN
    LN --> QF
    TOK --> QF
    QF --> PROJ

    TOK --> EMB
    PROJ --> CONCAT
    EMB --> CONCAT

    CONCAT --> LLAMA
    LLAMA --> WP
    LLAMA --> END

    WP --> PID
    END --> PID
    PID --> CTRL
```

---

## Simplified linear flow

```mermaid
flowchart LR
    A[Sensors] --> B[Visual encoder]
    B --> C[Q-Former]
    C --> D[llm_proj]
    D --> E[Concat text+image]
    E --> F[LLaMA]
    F --> G[Waypoints + End]
    G --> H[PID → Control]
```

---

## Component summary

| Block | Role |
|-------|------|
| **Visual encoder** | MemFuser: RGB + LiDAR + velocity → spatial features |
| **Q-Former** | Fuse instruction text + image; fixed-length query tokens |
| **llm_proj** | Map query tokens to LLM embedding space |
| **Concat** | Build sequence: [instruction] [frame1 img] [frame2 img] … |
| **LLaMA** | Process sequence; output = **hidden states** (no logits) |
| **Waypoints / End** | MLP heads on hidden states at frame positions → 5 waypoints, end prob |
| **PID** | Waypoints + end prob → steer, throttle, brake |

See [HIDDEN_STATES_AND_FLOW.md](HIDDEN_STATES_AND_FLOW.md) for detailed data flow and tensor shapes.
