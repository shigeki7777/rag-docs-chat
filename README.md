# RAG over your docs — case study

**Problem.** Teams want an AI that answers from *their* docs — without making things up.

**Solution.** A retrieval-augmented pipeline: ingest markdown → semantic embeddings (model2vec, CPU, no GPU) → cosine retrieve → **grounded answer with `[source]` citations**. If nothing relevant is found, it **refuses instead of hallucinating**.

**Verified run (real output):**
```
Q: what credit applies after 6h downtime this month?
A: Credit tiers by measured monthly uptime: 99.0%-99.9% -> 10% credit [sla.md]   (sim 0.63)
Q: is our customer data encrypted?
A: Encrypted at rest with AES-256 and in transit with TLS 1.2+ [security.md]      (sim 0.70)
Q: do you have an iPhone app?
A: I don't find that in the provided documentation.                              (sim 0.15 → refused)
```

**Why it sells:** clients fear hallucination. This shows grounding, citations, and a hard refusal threshold. Swap the in-memory store for pgvector and the generator for Claude — same interface. **Delivered in under a day.**
