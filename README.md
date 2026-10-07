# ml-concepts

Interactive SPAs that show how ML/LLM pieces work. Zero build: static files.

## transformers

One real model, run in the browser from its published weights. Every number on the page is computed live.

**Model: SmolLM2-135M**
- Llama architecture, Apache-2.0, 135M parameters
- 30 layers, d=576
- 9 heads sharing 3 key/value heads

**Eight stages**
1. **Embed:** tokens, their ids and the embedding-table rows, nearest tokens, and how other models add position here
2. **Attention:** all heads in a layer, a five-step walkthrough of one head (project, score, softmax, mix, combine), and any query/key pair followed through every layer and head
3. **Heads:** measured per-head behaviour (previous token, self, first token, focus, reach)
4. **Layer:** one layer in order: RMSNorm, attention, residual, SwiGLU MLP (gate, up, hidden), residual, plus a table of every layer
5. **Predict:** every position's next-token guess, and a logit lens by depth
6. **Space:** 3D view of tokens through the layers, every attention line, zoom/pan/reset
7. **Words:** 3D word map, and vector arithmetic searched over the whole vocabulary (zooms to the result and its nearest words)
8. **Generate:** one token at a time with a KV cache, temperature and top K

### Run it

```bash
python3 -m http.server 5391 -d transformers   # then open http://localhost:5391
node transformers/llama.test.mjs               # forward-pass invariants on synthetic weights
```

### Notes
- First run downloads 269 MB (bf16) from Hugging Face and keeps it in the browser's private file storage (OPFS; the Cache API rejects files this size).
- `src/llama.js` is our own forward pass (RoPE, GQA, RMSNorm, SwiGLU).
- It has not been compared numerically to a reference implementation yet. It is checked only for invariants, sensible predictions and working word analogies.
- Dependency: transformers.js 4.3.1 (tokenizer only, CDN-pinned).

## License

MIT, see [LICENSE](LICENSE). The SmolLM2-135M weights it downloads are separately licensed (Apache-2.0) by Hugging Face.
