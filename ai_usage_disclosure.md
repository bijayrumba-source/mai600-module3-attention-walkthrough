# AI Tool Usage Disclosure

**Assignment:** MAI 600 — Module 3, Attention Walkthrough
**Project:** Financial Fraud Alert Example

---

## AI tools used

- [ ] ChatGPT
- [x] **Claude**
- [ ] Claude Code
- [ ] GitHub Copilot
- [ ] OpenAI Codex
- [ ] Gemini
- [ ] Other: ___________

---

## How I used AI

- Explaining the query / key / value split in plainer language than the original paper uses, until I could describe each one's distinct job in my own words.
- Helping structure the notebook into clear sections (tokenization → attention mechanics → behavior documentation → reflection).
- Suggesting a layout for the annotated Transformer diagram so that all required components appeared in the correct order.
- Writing the scaled dot-product and multi-head attention functions in NumPy, and the sinusoidal positional encoding function.
- Improving grammar and clarity in the README.

---

## What I verified myself

- **Every number in this repository comes from actually running the notebook.** The token count (137 tokens from 123 words), the token positions of each reference word (`it` at 9, `they` at 29 and 75, `the activity` at 79, `although` at 108), and the subword splits (`dormant` → `d`/`orm`/`ant`) were all confirmed against real `tiktoken` output rather than assumed.
- I checked that the attention implementation is mathematically correct — specifically that every row of the attention weight matrix sums to 1.0 after the softmax, and that the tensor shapes are right at each step (`(n_heads, seq_len, seq_len)` for the weights).
- I confirmed the positional encoding claim by computing it: two identical token vectors have cosine similarity 1.0 before positional encoding and 0.873 after.
- I identified the five attention behaviors myself by reading the paragraph and asking which relationships could not be resolved from the words in isolation.

---

## What I changed or corrected

- I wrote the text sample myself rather than using a public dataset, so I could control exactly which reference relationships it contained and guarantee no sensitive data was involved.
- I added an explicit honesty note in the notebook clarifying that the from-scratch attention implementation uses **random, untrained** weights — so its output demonstrates the *mechanism* but is not linguistically meaningful. The attention behaviors are documented by linguistic annotation, not read off that toy model. I thought it would be misleading to present untrained attention output as if it showed real pronoun resolution.
- I noticed while building the token table that the two occurrences of `they` refer to different entities, and made that the centerpiece of the entity-tracking behavior rather than using a more routine example.
- I added the closing caution about attention weights not being a true explanation of model behavior, because the framing I encountered while researching seemed overconfident about what a heatmap actually proves.

---

*I confirm that I used AI as a learning and support tool, not as a replacement for my own work.*
