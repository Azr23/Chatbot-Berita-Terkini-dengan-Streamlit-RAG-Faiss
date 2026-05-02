# News Update Chatbot - RAG FAISS In-Memory

Aplikasi Streamlit ini adalah migrasi dari climate chatbot menjadi **chatbot news update** dengan arsitektur **RAG berbasis FAISS in-memory**. Tujuannya adalah membuat konteks yang dikirim ke LLM lebih kecil, lebih relevan, dan tetap punya sitasi sumber.

## Fitur Utama
- **RAG semantic retrieval** dengan FAISS in-memory (tanpa penyimpanan index ke disk)
- **Embedding lokal** menggunakan `sentence-transformers`
- **Whitelist domain sumber berita** (7 media)
- **Quick-load + manual refresh** sumber berita dari sidebar
- **URL custom** diperbolehkan selama masuk whitelist domain
- **Jawaban default Bahasa Indonesia** dengan sitasi sumber
- **Optimasi token prompt** dengan batas jumlah chunk dan excerpt

## Sumber Media (Whitelist)
1. `kompas.com`
2. `tempo.co`
3. `detik.com`
4. `cnnindonesia.com`
5. `kumparan.com`
6. `tirto.id`
7. `mediaindonesia.com`

Custom URL di luar domain di atas akan ditolak.

## Arsitektur Ringkas
1. Artikel di-fetch dari web dan dibersihkan dari boilerplate HTML.
2. Konten di-chunk dengan `RecursiveCharacterTextSplitter`.
3. Tiap chunk di-embed oleh model lokal multilingual.
4. Vektor dimasukkan ke FAISS `IndexFlatIP` in-memory.
5. Query user di-embed, lalu similarity search top-k.
6. Potongan konteks terbaik dikirim ke Gemini untuk jawaban bersitasi.

## Menjalankan Aplikasi

1. Install dependency:

```bash
pip install -r requirements.txt
```

2. Siapkan environment variable di `.env`:

```env
GEMINI_API_KEY=your_key_here
```

3. Jalankan Streamlit:

```bash
streamlit run app.py
```

atau

```bash
python -m streamlit run app.py
```


