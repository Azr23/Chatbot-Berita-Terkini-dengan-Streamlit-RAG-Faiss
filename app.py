import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from google import genai
from knowledge_base import NewsKnowledgeBase, NEWS_SOURCES, WHITELIST_DOMAINS

# Configure the Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

# Configure Streamlit page
st.set_page_config(
    page_title="News Update RAG FAISS",
    layout="wide"
)

MAX_CONTEXT_CHUNKS = 4
MAX_EXCERPT_CHARS = 700

# Initialize knowledge base in session state
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = NewsKnowledgeBase()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Halo! Saya asisten News Update berbasis RAG FAISS. Saya bisa menjawab berdasarkan artikel dari 7 media yang didukung dan memberikan sitasi sumber."
        }
    ]

def create_knowledge_enhanced_prompt(user_question: str, relevant_docs: list) -> str:
    """Create an Indonesian prompt with token-efficient RAG context."""

    context = "Tidak ada konteks dokumen relevan yang tersedia saat ini."
    sources_used = []

    if relevant_docs:
        context_lines = ["KONTEKS BERITA RELEVAN (RAG):"]
        for i, doc in enumerate(relevant_docs, 1):
            excerpt = (doc.get("content", "") or "")[:MAX_EXCERPT_CHARS]
            source = doc.get("source", "Unknown")
            url = doc.get("url", "-")
            score = doc.get("relevance_score", 0.0)
            context_lines.append(
                f"{i}. source={source} | url={url} | score={score}\nexcerpt: {excerpt}"
            )
            sources_used.append(doc['source'])
        context = "\n".join(context_lines)

    prompt = f"""Anda adalah asisten news update berbahasa Indonesia.
Jawab pertanyaan pengguna dengan mengutamakan konteks RAG yang tersedia.
Jika konteks tidak cukup, sampaikan keterbatasan dengan jujur dan tetap berikan jawaban aman.

ATURAN JAWABAN:
1. Gunakan Bahasa Indonesia sebagai default.
2. Ringkas, jelas, dan faktual.
3. Cantumkan sitasi sumber dalam format [Nama Sumber].
4. Jangan membuat klaim yang tidak didukung konteks.
5. Jika ada perbedaan sumber, sebutkan secara netral.

USER QUESTION: {user_question}
{context}
"""

    return prompt, sources_used

def display_knowledge_base_sidebar():
    """Display sidebar for whitelisted news ingestion and manual refresh."""
    with st.sidebar:
        st.markdown("## News RAG Control")

        stats = st.session_state.knowledge_base.get_document_stats()

        if stats["total_chunks"] > 0:
            st.success(
                f"Indexed {stats['indexed_vectors']} vectors dari {stats['total_chunks']} chunk dan {stats['sources']} sumber"
            )
            st.caption(f"Embedding model: {stats['embedding_model']}")
            with st.expander("Sumber aktif"):
                for source in stats["source_list"]:
                    st.write(f"- {source}")
        else:
            st.info("Belum ada chunk di index. Tambahkan sumber berita terlebih dahulu.")

        st.markdown("### Quick-load 7 Media")
        selected_source = st.selectbox(
            "Pilih media",
            [""] + list(NEWS_SOURCES.keys())
        )

        col_a, col_b = st.columns(2)
        if col_a.button("Load", use_container_width=True) and selected_source:
            with st.spinner(f"Load {selected_source}..."):
                url = NEWS_SOURCES[selected_source]
                documents = st.session_state.knowledge_base.load_web_article(url, selected_source)
                if documents:
                    st.success(f"Loaded {len(documents)} chunk dari {selected_source}")
                    st.rerun()
        if col_b.button("Refresh", use_container_width=True) and selected_source:
            with st.spinner(f"Refresh {selected_source}..."):
                url = NEWS_SOURCES[selected_source]
                documents = st.session_state.knowledge_base.refresh_web_article(url, selected_source)
                if documents:
                    st.success(f"Refresh berhasil: {len(documents)} chunk dari {selected_source}")
                    st.rerun()

        st.markdown("### URL Custom (Whitelist)")
        st.caption("Hanya domain yang diizinkan: " + ", ".join(sorted(WHITELIST_DOMAINS)))
        custom_url = st.text_input("Masukkan URL artikel")
        custom_name = st.text_input("Nama sumber (opsional)")
        custom_col_a, custom_col_b = st.columns(2)

        if custom_col_a.button("Load URL", use_container_width=True) and custom_url:
            with st.spinner("Load artikel custom..."):
                documents = st.session_state.knowledge_base.load_web_article(
                    custom_url,
                    custom_name if custom_name else None
                )
                if documents:
                    st.success(f"Loaded {len(documents)} chunk dari URL custom")
                    st.rerun()

        if custom_col_b.button("Refresh URL", use_container_width=True) and custom_url:
            with st.spinner("Refresh URL custom..."):
                documents = st.session_state.knowledge_base.refresh_web_article(
                    custom_url,
                    custom_name if custom_name else None,
                )
                if documents:
                    st.success(f"Refresh URL berhasil: {len(documents)} chunk")
                    st.rerun()

        # with st.expander("Upload PDF (sementara disembunyikan)", expanded=False):
        #     st.caption("Fitur PDF tidak dihapus permanen, namun disembunyikan pada fase ini.")

        if st.button("Clear Index", use_container_width=True):
            st.session_state.knowledge_base.clear()
            st.success("Index in-memory dibersihkan.")
            st.rerun()

def friendly_wrap_with_sources(raw_text: str, sources_used: list) -> str:
    """Friendly post-processing with compact source summary."""
    sources_section = ""
    if sources_used:
        unique_sources = list(set(sources_used))
        sources_section = "\n\nSumber yang digunakan:\n"
        for source in unique_sources:
            sources_section += f"- {source}\n"

    return (
        f"{raw_text.strip()}{sources_section}"
        "\nPerlu saya ringkas lagi, bandingkan antar media, atau fokus ke satu sumber tertentu?"
    )

def display_messages():
    """Display all messages in the chat"""
    for msg in st.session_state.messages:
        author = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(author):
            st.write(msg["content"])

# Create main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.title("News Update Chatbot")
    st.subheader("Chatbot Media Berita Terkini dengan RAG FAISS")

    # Display messages
    display_messages()

    # Handle new user input
    prompt = st.chat_input("Tanyakan update berita, isu hangat, atau minta ringkasan lintas sumber...")

    if prompt:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Show user message
        with st.chat_message("user"):
            st.write(prompt)

        # Show thinking indicator
        with st.chat_message("assistant"):
            placeholder = st.empty()

            # Search for relevant documents
            with st.spinner("Mencari chunk relevan di index..."):
                relevant_docs = st.session_state.knowledge_base.search_documents(prompt, max_results=MAX_CONTEXT_CHUNKS)

            if relevant_docs:
                placeholder.write("Menemukan konteks relevan, menyusun jawaban...")
            else:
                placeholder.write("Index kosong atau tidak ada konteks yang cocok. Menyusun jawaban fallback...")

            try:
                # Create enhanced prompt with document context
                enhanced_prompt, sources_used = create_knowledge_enhanced_prompt(prompt, relevant_docs)

                # Generate response with Gemini
                response = client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=enhanced_prompt
                )

                answer = response.text
                friendly_answer = friendly_wrap_with_sources(answer, sources_used)

            except Exception as e:
                friendly_answer = (
                    f"Terjadi error saat membuat jawaban: {e}. "
                    "Silakan coba lagi setelah memastikan API key Gemini sudah benar."
                )

            # Replace placeholder with actual response
            placeholder.write(friendly_answer)

            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": friendly_answer})

            # Show relevant document chunks if found
            if relevant_docs:
                with st.expander(f"Lihat {len(relevant_docs)} excerpt RAG"):
                    for i, doc in enumerate(relevant_docs, 1):
                        st.markdown(
                            f"**Source {i}: {doc.get('source', '-')}** "
                            f"(Score: {doc.get('relevance_score', 0.0)})"
                        )
                        st.caption(doc.get("url", "Tanpa URL"))
                        st.markdown(f"```\n{doc.get('content', '')[:350]}...\n```")
                        st.markdown("---")

with col2:
    display_knowledge_base_sidebar()

# Footer
st.markdown("---")
stats = st.session_state.knowledge_base.get_document_stats()
# st.markdown(
#     f"Status index: {stats['indexed_vectors']} vectors | "
#     f"{stats['total_chunks']} chunk | {stats['sources']} sumber"
# )