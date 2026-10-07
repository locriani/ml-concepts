# ml-concepts

Interactive SPAs that show how ML/LLM pieces work. Zero build: static files.

## transformers

One real model, SmolLM2-135M (Llama architecture, Apache-2.0, 135M params, 30 layers, 9 heads sharing 3 key/value heads, d=576), run in the browser from its published weights. Eight stages: Embed, Attention (all heads, then any query/key pair followed through every layer and head), Heads (measured per-head behaviour), Layer (RMSNorm, attention, SwiGLU MLP, residuals), Predict (every position's next-token guess and a logit lens by depth), Space (3D, every attention line, zoom/pan/reset), Words (3D word map and vector arithmetic searched over the whole vocabulary), Generate (one token at a time with KV cache, temperature and top K).

```bash
python3 -m http.server 5391 -d transformers   # then open http://localhost:5391
node transformers/llama.test.mjs               # forward-pass invariants on synthetic weights
npx -y firebase-tools@15.32.1 deploy --only hosting   # publish to https://ml-concepts.letstakethis.online (Firebase project canwetakethisonline, needs `firebase login`)
```

First run downloads 269 MB (bf16) from Hugging Face and keeps it in the browser's private file storage (OPFS; the Cache API rejects files this size). `src/llama.js` is our own forward pass (RoPE, GQA, RMSNorm, SwiGLU); it has not been compared numerically to a reference implementation yet, only checked for invariants, sensible predictions and working word analogies. Dependency: transformers.js 4.3.1 (tokenizer only, CDN-pinned).

## License

MIT, see [LICENSE](LICENSE). The SmolLM2-135M weights it downloads are separately licensed (Apache-2.0) by Hugging Face.
