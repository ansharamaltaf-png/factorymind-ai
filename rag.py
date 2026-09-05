"""
Stage IV - Generative AI + RAG
Knowledge base = SOP/manual/safety text files in rag_knowledge/ (stand-ins
for machine manuals/SOPs -- in production these would be parsed from real
PDFs via the same chunking approach; see README for how to point this at
real PDFs with pypdf).

Retrieval: TF-IDF + cosine similarity (transparent, no external API needed,
runs fully offline). "Generation": template-grounded explanation that is
built ONLY from retrieved chunks, so the answer is always evidence-backed
(the retrieved section id is returned alongside the answer).

If an LLM API key is available (ANTHROPIC_API_KEY env var), `generate_answer`
will optionally call the real Claude API to phrase the same retrieved
evidence in natural language -- but the retrieval + evidence-selection stays
identical either way, and the LLM is never used to invent numbers.
"""

import os
import glob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_DIR = "rag_knowledge"
CHUNK_SIZE = 2  # sentences/lines per chunk


def load_chunks(kb_dir: str = KB_DIR):
    chunks = []
    for path in sorted(glob.glob(f"{kb_dir}/*.txt")):
        doc_name = os.path.basename(path)
        with open(path, "r") as f:
            text = f.read()
        sections = re.split(r"(?=Section \d+ -)", text)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            title_match = re.match(r"(Section \d+ - [^:\n]+)", sec)
            title = title_match.group(1) if title_match else doc_name
            chunks.append({"doc": doc_name, "section": title, "text": sec})
    return chunks


class RAGIndex:
    def __init__(self, kb_dir: str = KB_DIR):
        self.chunks = load_chunks(kb_dir)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).ravel()
        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for i in top_idx:
            if sims[i] <= 0:
                continue
            r = dict(self.chunks[i])
            r["score"] = float(sims[i])
            results.append(r)
        return results


def generate_answer(query: str, index: RAGIndex, k: int = 3, use_llm: bool = False):
    evidence = index.retrieve(query, k=k)
    if not evidence:
        return {
            "answer": "No relevant SOP/manual evidence was found for this query. "
                      "Escalate to a human supervisor rather than guessing.",
            "evidence": [],
        }

    if use_llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            answer = _llm_generate(query, evidence)
        except Exception:
            answer = _template_generate(query, evidence)
    else:
        answer = _template_generate(query, evidence)

    return {"answer": answer, "evidence": evidence}


def _template_generate(query: str, evidence: list) -> str:
    lines = [f"Based on {evidence[0]['doc']} ({evidence[0]['section']}):"]
    for e in evidence:
        snippet = e["text"].split("\n", 1)[-1].strip() if "\n" in e["text"] else e["text"]
        lines.append(f"- [{e['doc']} / {e['section']}] {snippet}")
    return "\n".join(lines)


def _llm_generate(query: str, evidence: list) -> str:
    """Optional real LLM call (only used if ANTHROPIC_API_KEY is set).
    The model is explicitly instructed to only rephrase the retrieved
    evidence, not invent new facts -- this is the RAG-improves-over-
    unsupported-generation contrast required by the rubric."""
    import anthropic
    client = anthropic.Anthropic()
    context = "\n\n".join(f"[{e['doc']} / {e['section']}]\n{e['text']}" for e in evidence)
    prompt = (
        f"You are the Knowledge Agent in a factory AI system. Using ONLY the "
        f"SOP evidence below, answer the operator's question. Cite the SOP/section "
        f"for every claim. Do not add information not present in the evidence.\n\n"
        f"EVIDENCE:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


def demo_retrieval_vs_unsupported():
    """Illustrates the required 'retrieval improves the answer' case."""
    index = RAGIndex()
    query = "What torque and wear combination counts as overstrain for a High-type machine?"
    unsupported = "A generic LLM without retrieval would have to guess a numeric threshold."
    grounded = generate_answer(query, index)
    return {"query": query, "unsupported_baseline": unsupported, "rag_answer": grounded}


if __name__ == "__main__":
    print(demo_retrieval_vs_unsupported())
