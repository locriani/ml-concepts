# ml-concepts

Interactive SPAs that show how ML/LLM pieces work. Zero build: static files.

## transformers

The model runs in the browser from its published weights. Every number on the page is computed live.

**Model: SmolLM2-135M**
- Llama architecture with 135M parameters under Apache-2.0
- 30 layers with d=576
- 9 heads sharing 3 key/value heads

**Eight stages**
1. **Embed:** tokens and token ids and embedding table rows. Nearest tokens. Position encoding comparisons across other models.
2. **Attention:** all heads in a layer. A five-step walkthrough of one head covering project then score then softmax then mix then combine. Tracking for any query-key pair through every layer and head.
3. **Heads:** measured per-head behaviour covering previous token and self and first token and focus and reach.
4. **Layer:** one layer in order through RMSNorm and attention and residual and SwiGLU MLP (gate and up and hidden) and residual. A table of every layer.
5. **Predict:** next-token predictions at every position. A logit lens by depth.
6. **Space:** 3D view of tokens across layers. Attention lines between tokens. Controls for zoom and pan and reset.
7. **Words:** 3D word map. Vector arithmetic across the full vocabulary with focus on results and nearest words.
8. **Generate:** token generation with a KV cache plus temperature and top K sampling.

### Run it

```bash
python3 -m http.server 5391 -d transformers   # local server at http://localhost:5391
node transformers/llama.test.mjs               # forward-pass invariants on synthetic weights
npx -y firebase-tools@15.32.1 deploy --only hosting   # publish to https://ml-concepts.canwetakethis.online (Firebase project canwetakethisonline requiring firebase login)
```

### Notes
- First run downloads 269 MB (bf16) from Hugging Face and keeps it in the browser's private file storage (OPFS; the Cache API rejects files this size).
- `src/llama.js` implements the forward pass with RoPE and GQA and RMSNorm and SwiGLU.
- It has not been compared numerically to a reference implementation yet. It is checked for invariants and sensible predictions and working word analogies.
- Dependency: transformers.js 4.3.1 (tokenizer only with pinned CDN URL).

## License

MIT license. See [LICENSE](LICENSE). The downloaded SmolLM2-135M weights carry an Apache-2.0 license from Hugging Face.
