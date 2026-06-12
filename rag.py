"""
RAG over your docs — grounded answers with citations, no hallucination.

Pipeline:  ingest .md -> sentence chunks -> semantic embeddings (model2vec, CPU, no torch)
           -> cosine retrieve -> grounded answer with [source] citation
           -> if best similarity < threshold: refuse ("not in the documentation").

The GROUNDING_PROMPT turns the retrieved, cited context into conversational prose via
Claude / any LLM — swap the generator without touching retrieval. For larger corpora,
swap the in-memory store for pgvector / a vector DB; the interface stays the same.
"""
import re, glob, os
import numpy as np
from model2vec import StaticModel

_MODEL = "minishlab/potion-base-8M"

def sentences(text):
    # drop markdown heading lines first, THEN sentence-split (keeps each doc's first fact)
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
    return [s.strip() for s in re.split(r'(?<=[.!?;])\s+', body) if s.strip()]

class RAG:
    def __init__(self, folder, model_name=_MODEL):
        self.model = StaticModel.from_pretrained(model_name)
        self.chunks = []
        for f in sorted(glob.glob(os.path.join(folder, "*.md"))):
            name = os.path.basename(f)
            for s in sentences(open(f).read()):
                self.chunks.append((name, s))
        E = self.model.encode([c[1] for c in self.chunks]).astype("float32")
        self.E = E / np.linalg.norm(E, axis=1, keepdims=True)

    def retrieve(self, query, k=3):
        q = self.model.encode([query]).astype("float32")[0]
        q = q / (np.linalg.norm(q) or 1.0)
        sims = self.E @ q
        idx = np.argsort(-sims)[:k]
        return [(float(sims[i]), self.chunks[i][0], self.chunks[i][1]) for i in idx]

    def answer(self, query, threshold=0.30):
        hits = self.retrieve(query)
        if not hits or hits[0][0] < threshold:
            return {"answer": "I don't find that in the provided documentation.",
                    "grounded": False, "sources": [], "score": round(hits[0][0], 3) if hits else 0.0}
        keep = [h for h in hits if h[0] >= max(threshold, 0.75 * hits[0][0])][:2]
        ans = " ".join(f"{t} [{s}]" for _, s, t in keep)
        return {"answer": ans, "grounded": True,
                "sources": sorted({s for _, s, _ in keep}), "score": round(hits[0][0], 3)}

GROUNDING_PROMPT = ("Answer the QUESTION using ONLY the CONTEXT. Cite each fact as [source]. "
                    "If the answer is not in the CONTEXT, say you don't find it in the documentation. Be concise.")

if __name__ == "__main__":
    rag = RAG(os.path.join(os.path.dirname(__file__), "docs"))
    print(f"ingested {len(rag.chunks)} sentence-chunks from "
          f"{len(set(c[0] for c in rag.chunks))} docs (semantic embeddings)\n")
    for q in ["We had 6 hours of downtime this month, what service credit applies?",
              "Is our customer data encrypted?",
              "Can I get a refund after two months?",
              "Do you have an iPhone app?"]:
        r = rag.answer(q)
        print("Q:", q)
        print("A:", r["answer"])
        print(f"   grounded={r['grounded']}  score={r['score']}  sources={r['sources']}\n")
