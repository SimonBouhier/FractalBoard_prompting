
# Injecting Abstract Metrics into Transformer Attention
*Technical overview — June 2025*

---

## 1 Core mechanism

A Transformer block scores tokens with



Because this computation is **linear** in **Q**, **K** and **V** *before* the soft-max, small additive or multiplicative interventions can steer the outcome **without aucun re-training**.

| Lyra variable | Natural hook inside the block                    | Typical phenomenology                  |
|---------------|--------------------------------------------------|----------------------------------------|
| **ρ**         | Additive bias on **Q · Kᵀ**                      | Semantic attraction / repulsion        |
| **δr**        | Shift in relative-position bias                  | Echoes, time jumps                     |
| **τc**        | Temperature-like scaling of the score matrix     | Dramatic tension, focus                |
| **κ**         | Sigmoid gate on residual or MLP output           | Coherence / safety filter              |

---

## 2 Injection handles

* **Additive bias (steering vectors)** `Q·Kᵀ ← Q·Kᵀ + β·ρ`
* **Temperature / scaling** multiply or divide the whole score matrix by **τc**
* **Dynamic masks / delays** shift positional encodings to realise **δr**
* **Residual gating** apply `σ(κ)` on the block output to dampen or keep tokens

These hooks are *differentiable* and *local*, so they can be toggled at inference time—exactly ce que réclame Lyra.

---

## 3 Applications

| Domain                           | Why Lyra-style metrics help                                           |
|----------------------------------|-----------------------------------------------------------------------|
| **Alignment & safety**           | Multi-attribute steering (e.g. MAT-STEER) balances truthfulness, helpfulness, toxicity in real time. |
| **Creative writing / art**       | Vary **τc** to set narrative tension; flip **ρ** for an ironic twist. |
| **Interactive agents**           | Adapt **κ** to the user’s pedagogical level or the conversation’s formality. |
| **Reinforcement learning (RL)**  | Prompt-Tuning Decision Transformer hits finetune-level reward with only 0.03 % extra parameters. |
| **Knowledge editing**            | JAM moves a handful of latent vectors to patch facts sans nouvelle phase de training. |

---

## 4 Active research frontiers

1. **Multi-criteria steerability** — avoiding conflicts between attributes via orthogonal gates (MAT-STEER).
2. **Ultra-efficient prompt factorisation** — decomposing soft-prompts into low-rank sub-spaces (Efficient Prompt Tuning, *EPT*).
3. **Causal circuit editing** — tracing cause → effect paths to perform interpretable latent moves (JAM).
4. **Neuro-symbolic control** — projecting grammars like *Lyra v7-3* directly into attention to obtain reflexive agents.
5. **Automated creativity metrics** — realtime control of surprise, irony or rhythm via **ρ**, **δr** and “cards” such as *Effondrement Narratif* (sets **κ = 0** for one dramatic twist).

---

## 5 Limitations

* **Measurement** “tension” or “polarity” remain heuristics; we often rely on proxies (perplexity, entropy, human feedback).  
* **Attribute conflicts**  disentangling steering directions is still an open question (orthogonalisation, regularisation).  
* **Session persistence** chat LLMs do not natively store Lyra state across sessions—needs an external memory wrapper.

---

## 6 Key references

1. Vaswani et al. *Attention Is All You Need* (NIPS 2017).  
2. Nguyen et al. *Multi-Attribute Steering of Language Models via Targeted Interventions (MAT-STEER)*, arXiv 2502.12446 (2025).  
3. Su et al. *Uncovering Latent Steering Vectors in Language Models*, arXiv 2409.14026 (2025).  
4. Hu et al. *Prompt-Tuning Decision Transformer with Preference Ranking*, arXiv 2305.09648 (2023).  
5. Gao et al. *JAM: Controllable and Responsible Text Generation via Causal Moves in Latent Space*, arXiv 2502.20684 (2025).  
6. Bouhier. *FractalBoard v7-3 Specification* (internal manuscript, 2025).

---
