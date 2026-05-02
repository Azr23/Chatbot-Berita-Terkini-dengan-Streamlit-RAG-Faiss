# knowledge_base.py
import os
import re
import hashlib
from typing import Dict, List, Optional
from urllib.parse import urlparse

import faiss
import numpy as np
import requests
import streamlit as st
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


NEWS_SOURCES = {
    "Kompas.com": "https://www.kompas.com/",
    "Tempo.co": "https://www.tempo.co/",
    "Detik.com": "https://www.detik.com/",
    "CNNIndonesia.com": "https://www.cnnindonesia.com/",
    "Kumparan.com": "https://kumparan.com/",
    "Tirto.id": "https://tirto.id/",
    "Mediaindonesia.com": "https://mediaindonesia.com/",
}

WHITELIST_DOMAINS = {
    "kompas.com",
    "tempo.co",
    "detik.com",
    "cnnindonesia.com",
    "kumparan.com",
    "tirto.id",
    "mediaindonesia.com",
}


class NewsKnowledgeBase:
    def __init__(self):
        self.documents: List[Dict] = []
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=180,
            length_function=len,
        )
        self.embedding_model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

        sample_vector = self.embedding_model.encode(["inisialisasi"], normalize_embeddings=True)
        self.embedding_dim = int(sample_vector.shape[1])
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    @staticmethod
    def _normalize_domain(domain: str) -> str:
        domain = (domain or "").lower().strip()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    def _is_allowed_domain(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = self._normalize_domain(parsed.netloc)
        if not domain:
            return False
        return any(domain == allowed or domain.endswith(f".{allowed}") for allowed in WHITELIST_DOMAINS)

    @staticmethod
    def _clean_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "")
        return cleaned.strip()

    def _extract_headline_fallback(self, soup: BeautifulSoup, max_items: int = 80) -> str:
        """Aggregate headline-like text when full article body is not available."""
        lines: List[str] = []
        seen = set()

        def add_line(value: str) -> None:
            text = self._clean_text(value)
            if not text:
                return
            low = text.lower()
            if low in seen:
                return
            # Basic noise guard for nav/menu fragments.
            if len(text) < 20 or len(text) > 240:
                return
            if len(text.split()) < 4:
                return
            if any(token in low for token in ["login", "register", "menu", "privacy", "copyright", "terms"]):
                return
            seen.add(low)
            lines.append(text)

        for meta_name in ["description", "og:description", "twitter:description"]:
            meta = soup.find("meta", attrs={"name": meta_name}) or soup.find("meta", attrs={"property": meta_name})
            if meta and hasattr(meta, "get"):
                add_line(meta.get("content") or "")

        for title_tag in soup.find_all(["h1", "h2", "h3", "h4"]):
            if len(lines) >= max_items:
                break
            add_line(title_tag.get_text(" ", strip=True))

        if not lines:
            return ""
        return self._clean_text("\n".join(lines))

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        vectors = self.embedding_model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)

    @staticmethod
    def _score_to_float(score: float) -> float:
        return float(round(score, 4))

    def _append_documents(self, docs: List[Dict]) -> None:
        if not docs:
            return
        vectors = self._embed_texts([doc["content"] for doc in docs])
        self.index.add(vectors)
        self.documents.extend(docs)

    def _extract_article_text(self, html_content: bytes) -> str:
        """Extract article text from HTML content using defensive fallback approach."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove non-content elements by tag name only (safe, no attribute assumptions)
        for element in soup.find_all(["script", "style", "noscript", "header", "footer", "aside", "nav", "form"]):
            try:
                element.decompose()
            except Exception:
                pass  # Ignore if element can't be removed

        # Use <article> only when it looks like a real full article body.
        article_candidate = soup.find("article")
        if article_candidate is not None:
            try:
                article_text = self._clean_text(article_candidate.get_text(" ", strip=True))
                if len(article_text) >= 400:
                    return article_text
            except (AttributeError, TypeError):
                pass

        # Homepage/listing pages often work better with main/body aggregation.
        candidate = soup.find("main") or soup.body or soup
        try:
            raw_text = candidate.get_text(" ", strip=True)
        except (AttributeError, TypeError):
            raw_text = soup.get_text(" ", strip=True)

        headline_text = self._extract_headline_fallback(soup)
        cleaned = self._clean_text(raw_text)

        if len(cleaned) >= 200:
            # If body text is dominated by nav/menu boilerplate, prefer headline fallback.
            body_preview = f" {cleaned[:1400].lower()} "
            noise_markers = [
                " menu ",
                " kategori berita ",
                " login ",
                " register ",
                " iklan ",
                " artikel trending ",
                " top up ",
                " detik network ",
            ]
            noise_hits = sum(1 for marker in noise_markers if marker in body_preview)
            if headline_text and len(headline_text) >= 120 and noise_hits >= 2:
                return headline_text
            return cleaned

        # Last fallback for JS-heavy pages: aggregate headline-like snippets.
        if headline_text:
            return headline_text
        return cleaned

    def _chunk_to_documents(
        self,
        text: str,
        source_name: str,
        source_type: str,
        url: Optional[str] = None,
        page_count: Optional[int] = None,
    ) -> List[Dict]:
        chunks = self.text_splitter.split_text(text)
        documents: List[Dict] = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            content_hash = hashlib.md5(f"{source_name}-{i}-{chunk[:120]}".encode("utf-8")).hexdigest()[:12]
            doc = {
                "content": chunk,
                "source": source_name,
                "source_type": source_type,
                "chunk_id": f"{source_name}_{i}_{content_hash}",
            }
            if url:
                doc["url"] = url
            if page_count is not None:
                doc["page_count"] = page_count
            documents.append(doc)

        return documents

    def remove_source(self, source_name: str, url: Optional[str] = None) -> int:
        original_count = len(self.documents)
        remaining_docs = []
        for doc in self.documents:
            is_same_source = doc.get("source") == source_name
            is_same_url = (url is None) or (doc.get("url") == url)
            if is_same_source and is_same_url:
                continue
            remaining_docs.append(doc)

        removed = original_count - len(remaining_docs)
        if removed == 0:
            return 0

        self.documents = remaining_docs
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        if self.documents:
            vectors = self._embed_texts([doc["content"] for doc in self.documents])
            self.index.add(vectors)
        return removed

    def clear(self) -> None:
        self.documents = []
        self.index = faiss.IndexFlatIP(self.embedding_dim)

    def load_pdf(self, file_path: str, source_name: Optional[str] = None) -> List[Dict]:
        """Load and index a PDF file into the in-memory FAISS index."""
        try:
            if source_name is None:
                source_name = os.path.basename(file_path)

            reader = PdfReader(file_path)
            full_text_parts = []
            for page in reader.pages:
                full_text_parts.append(page.extract_text() or "")

            full_text = self._clean_text("\n".join(full_text_parts))
            if not full_text:
                return []

            docs = self._chunk_to_documents(
                text=full_text,
                source_name=source_name,
                source_type="PDF",
                page_count=len(reader.pages),
            )
            self._append_documents(docs)
            return docs
        except Exception as exc:
            st.error(f"Error loading PDF {file_path}: {exc}")
            return []

    def load_web_article(self, url: str, source_name: Optional[str] = None) -> List[Dict]:
        """Load and index one web article from a whitelisted news domain."""
        try:
            if not self._is_allowed_domain(url):
                st.error("Domain URL tidak ada di whitelist sumber berita.")
                return []

            parsed = urlparse(url)
            if source_name is None:
                source_name = self._normalize_domain(parsed.netloc)

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()

            text = self._extract_article_text(response.content)
            if len(text) < 120:
                st.warning("Konten artikel terlalu pendek atau tidak terbaca setelah dibersihkan.")
                return []

            docs = self._chunk_to_documents(
                text=text,
                source_name=source_name,
                source_type="Web Article",
                url=url,
            )
            self._append_documents(docs)
            return docs
        except Exception as exc:
            st.error(f"Error loading web article {url}: {exc}")
            return []

    def refresh_web_article(self, url: str, source_name: Optional[str] = None) -> List[Dict]:
        parsed = urlparse(url)
        resolved_source = source_name or self._normalize_domain(parsed.netloc)
        self.remove_source(resolved_source, url=url)
        return self.load_web_article(url=url, source_name=resolved_source)

    def search_documents(self, query: str, max_results: int = 4) -> List[Dict]:
        """Semantic retrieval with FAISS cosine similarity."""
        if not query.strip() or not self.documents or self.index.ntotal == 0:
            return []

        query_vector = self._embed_texts([query])
        top_k = min(max_results, len(self.documents))
        scores, indices = self.index.search(query_vector, top_k)

        results: List[Dict] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            doc = self.documents[int(idx)].copy()
            doc["relevance_score"] = self._score_to_float(float(score))
            results.append(doc)

        return results

    def get_document_stats(self) -> Dict:
        """Get statistics for loaded chunks and FAISS index status."""
        if not self.documents:
            return {
                "total_chunks": 0,
                "sources": 0,
                "source_list": [],
                "types": [],
                "indexed_vectors": 0,
                "embedding_model": self.embedding_model_name,
            }

        sources = sorted(set(doc["source"] for doc in self.documents))
        types = sorted(set(doc["source_type"] for doc in self.documents))
        return {
            "total_chunks": len(self.documents),
            "sources": len(sources),
            "source_list": sources,
            "types": types,
            "indexed_vectors": int(self.index.ntotal),
            "embedding_model": self.embedding_model_name,
        }


# Backward compatibility alias for existing imports.
ClimateKnowledgeBase = NewsKnowledgeBase
CLIMATE_SOURCES = NEWS_SOURCES