# ml-concepts

Interactive SPAs that show how ML/LLM pieces work. Zero build: static files.

## transformers

The model runs in the browser from its published weights. Every number on the page is computed live.

**Model: SmolLM2-135M**
- Llama architecture with 135M parameters under Apache-2.0
- 30 layers with d=576
- 9 heads sharing 3 key/value heads

**Eight stages**
1. **Embed:** tokens, token ids, embedding table rows. Nearest tokens. Position encoding comparisons across other models.
2. **Attention:** all heads in a layer. A five-step walkthrough of one head covering project then score then softmax then mix then combine. Tracking for any query-key pair through every layer and head.
3. **Heads:** measured per-head behaviour covering previous token, self, first token, focus and reach.
4. **Layer:** one layer in order through RMSNorm, attention, residual, SwiGLU MLP (gate, up, hidden) and residual. A table of every layer.
5. **Predict:** next-token predictions at every position. A logit lens by depth.
6. **Space:** 3D view of tokens across layers. Attention lines between tokens. Controls for zoom, pan and reset.
7. **Words:** 3D word map. Vector arithmetic across the full vocabulary with focus on results and nearest words.
8. **Generate:** token generation with a KV cache plus temperature and top K sampling.

### Run it

```bash
python3 -m http.server 5391 -d transformers   # local server at http://localhost:5391
node transformers/llama.test.mjs               # forward-pass invariants on synthetic weights
node transformers/learn/tiny.test.mjs          # backward pass against finite differences
node transformers/queue/sim.test.mjs           # M/M/1 means, Little's law, unstable growth
npx -y firebase-tools@15.32.1 deploy --only hosting   # publish to https://ml-concepts.canwetakethis.online (Firebase project canwetakethisonline requiring firebase login)
```

### Notes
- First run downloads 269 MB (bf16) from Hugging Face and keeps it in the browser's private file storage (OPFS; the Cache API rejects files this size).
- `src/llama.js` implements the forward pass with RoPE, GQA, RMSNorm and SwiGLU.
- It has not been compared numerically to a reference implementation yet. It is checked for invariants, sensible predictions and working word analogies.
- Dependency: transformers.js 4.3.1 (tokenizer only with pinned CDN URL).

## learning

A separate series from the transformer explainer. A small model guesses the next word of a few real sentences, and you can add your own. The bars are the probability of each true next word. The height is the loss. Training steps follow the gradient, scaled by the learning rate. One sentence’s gradient leaves the others short. The mean gradient fits the set. The tabs are backpropagation from the logits to the embedding.

The transformer server also serves this series at <http://localhost:5391/learn/>.

## queues

A supermarket with several registers, and the same people in two layouts. A line at each register, or one line that feeds the next free register. Utilization ρ = λ/(cμ) is the fraction of time each register is busy, and it is the same in either layout. The wait is not. Separate lines are c copies of an M/M/1 queue. One shared line is an M/M/c queue, and its wait uses Erlang's C formula. Each symbol in the formula highlights the part of the store it names.

Served at <http://localhost:5391/queue/>.

## License

MIT license. See [LICENSE](LICENSE). The downloaded SmolLM2-135M weights carry an Apache-2.0 license from Hugging Face.
