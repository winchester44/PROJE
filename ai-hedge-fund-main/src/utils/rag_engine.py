"""
RAG (Retrieval-Augmented Generation) Motoru
============================================
Yerel PDF/TXT dosyalarını (bilanço raporları, KAP bildirimleri, yıllık raporlar)
okuyup vektör veritabanına (Chroma) yükler.  LLM ajanlarının promptlarına
derin-okuma bağlamı (context) olarak sunar.

Kullanım:
    from src.utils.rag_engine import RAGEngine

    rag = RAGEngine(docs_dir="docs/reports")
    rag.ingest()   # İlk yükleme (sadece bir kez çağırmanız yeterli)
    context = rag.query("THYAO brüt kâr marjı nedir?")
"""

from __future__ import annotations

import os
import glob
from typing import Optional

# ── Lazy import wrappers ──────────────────────────────────────────────────────
# Tüm ağır bağımlılıklar (LangChain, Chroma, PyPDF2) ihtiyaç anında yüklenir
# böylece RAG kullanılmadığında başlangıç süresi etkilenmez.

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DOCS_DIR = os.path.join(_BASE_DIR, "..", "..", "docs")
_DEFAULT_PERSIST_DIR = os.path.join(_BASE_DIR, "..", "..", ".chroma_db")


class RAGEngine:
    """
    PDF / TXT dosyalarını vektör veritabanında depolar ve
    soru-cevap sorgularına en alakalı bölümleri (chunk) döner.

    Parameters
    ----------
    docs_dir : str
        PDF ve TXT dosyalarının bulunduğu dizin.
    persist_dir : str
        Chroma veritabanının kalıcı depolama dizini.
    chunk_size : int
        Metin parçalama boyutu (karakter).
    chunk_overlap : int
        Parçalar arası örtüşme (karakter).
    embedding_model : str
        HuggingFace embedding modeli adı. Varsayılan: sentence-transformers/all-MiniLM-L6-v2
    """

    def __init__(
        self,
        docs_dir: str = _DEFAULT_DOCS_DIR,
        persist_dir: str = _DEFAULT_PERSIST_DIR,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.docs_dir = docs_dir
        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model

        self._vectorstore = None
        self._embeddings = None

    # ──────────────────────────────────────────────────────────────────────────
    # LAZY INIT
    # ──────────────────────────────────────────────────────────────────────────
    def _get_embeddings(self):
        """HuggingFace embedding modelini lazily yükler."""
        if self._embeddings is None:
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embeddings

    def _get_vectorstore(self):
        """Chroma vektör deposunu yükler veya oluşturur."""
        if self._vectorstore is None:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma
            self._vectorstore = Chroma(
                collection_name="financial_docs",
                embedding_function=self._get_embeddings(),
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    # ──────────────────────────────────────────────────────────────────────────
    # DOCUMENT LOADING
    # ──────────────────────────────────────────────────────────────────────────
    def _extract_ticker_from_filename(self, filename: str) -> Optional[str]:
        """Dosya adından hisse sembolünü (ticker) ayıklar."""
        if "_internet_raporu" in filename:
            return filename.split("_internet_raporu")[0].upper()
        if "_hot_data" in filename:
            return filename.split("_hot_data")[0].upper()
        if "_" in filename:
            return filename.split("_")[0].upper()
        return None

    def _load_documents(self) -> list:
        """docs_dir altındaki PDF ve TXT dosyalarını yükler."""
        from langchain_core.documents import Document

        documents: list[Document] = []

        if not os.path.isdir(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            return documents

        # PDF Dosyaları
        pdf_files = glob.glob(os.path.join(self.docs_dir, "**", "*.pdf"), recursive=True)
        for pdf_path in pdf_files:
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(pdf_path)
                text_pages = []
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_pages.append(page_text)
                if text_pages:
                    full_text = "\n\n".join(text_pages)
                    filename = os.path.basename(pdf_path)
                    ticker = self._extract_ticker_from_filename(filename)
                    
                    metadata = {
                        "source": filename,
                        "type": "pdf",
                        "path": pdf_path,
                    }
                    if ticker:
                        metadata["ticker"] = ticker
                        
                    documents.append(
                        Document(
                            page_content=full_text,
                            metadata=metadata,
                        )
                    )
            except Exception as e:
                print(f"[RAG] PDF okunamadı ({pdf_path}): {e}")

        # TXT / Markdown Dosyaları
        for ext in ("*.txt", "*.md"):
            txt_files = glob.glob(os.path.join(self.docs_dir, "**", ext), recursive=True)
            for txt_path in txt_files:
                try:
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        filename = os.path.basename(txt_path)
                        ticker = self._extract_ticker_from_filename(filename)
                        
                        metadata = {
                            "source": filename,
                            "type": "text",
                            "path": txt_path,
                        }
                        if ticker:
                            metadata["ticker"] = ticker
                            
                        documents.append(
                            Document(
                                page_content=content,
                                metadata=metadata,
                            )
                        )
                except Exception as e:
                    print(f"[RAG] Dosya okunamadı ({txt_path}): {e}")

        return documents

    # ──────────────────────────────────────────────────────────────────────────
    # INGESTION (Vektör Deposuna Yükleme)
    # ──────────────────────────────────────────────────────────────────────────
    def ingest(self) -> int:
        """
        Dokümanları yükler, chunk'lara böler ve vektör deposuna ekler.
        Mevcut dokümanların eski chunk'larını yinelenen kayıtları önlemek için temizler.

        Returns
        -------
        int
            Eklenen toplam chunk sayısı.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        documents = self._load_documents()
        if not documents:
            print("[RAG] docs/ dizininde yüklenecek dosya bulunamadı.")
            return 0

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            return 0

        vs = self._get_vectorstore()

        # Önce mevcut dokümanların eski chunk'larını temizle (yinelenen kayıtları önlemek için)
        try:
            for doc in documents:
                source_name = doc.metadata.get("source")
                if source_name:
                    vs._collection.delete(where={"source": source_name})
        except Exception as e:
            print(f"[RAG] Eski chunk temizleme hatası: {e}")

        vs.add_documents(chunks)
        print(f"[RAG] {len(chunks)} chunk vektör deposuna eklendi ({len(documents)} dosya).")
        return len(chunks)

    # ──────────────────────────────────────────────────────────────────────────
    # QUERY (Sorgulama)
    # ──────────────────────────────────────────────────────────────────────────
    def query(self, question: str, top_k: int = 4, ticker: Optional[str] = None) -> str:
        """
        Soruyu vektör deposunda arar, en alakalı chunk'ları birleştirip döner.

        Parameters
        ----------
        question : str
            Doğal dildeki soru (örn. "THYAO'nun son çeyrek brüt kâr marjı nedir?")
        top_k : int
            Döndürülecek maksimum chunk sayısı.
        ticker : Optional[str]
            Aramayı belirli bir hisse senediyle filtreler.

        Returns
        -------
        str
            Birleştirilmiş bağlam metni. Sonuç yoksa boş string.
        """
        try:
            vs = self._get_vectorstore()

            # Koleksiyon boşsa sorgu yapmaya gerek yok
            if vs._collection.count() == 0:
                return ""

            filter_dict = None
            if ticker:
                filter_dict = {"ticker": ticker.upper()}

            results = vs.similarity_search(question, k=top_k, filter=filter_dict)

            if not results:
                return ""

            context_parts = []
            for i, doc in enumerate(results, 1):
                source = doc.metadata.get("source", "bilinmeyen")
                context_parts.append(
                    f"[Kaynak {i}: {source}]\n{doc.page_content}"
                )

            return "\n\n---\n\n".join(context_parts)

        except Exception as e:
            print(f"[RAG] Sorgu hatası: {e}")
            return ""

    def has_documents(self) -> bool:
        """Vektör deposunda doküman olup olmadığını kontrol eder."""
        try:
            vs = self._get_vectorstore()
            return vs._collection.count() > 0
        except Exception:
            return False


# ── Modül seviyesi kolay-kullanım fonksiyonları ───────────────────────────────

_default_rag: Optional[RAGEngine] = None


def get_rag_engine(**kwargs) -> RAGEngine:
    """Singleton RAG motoru döner."""
    global _default_rag
    if _default_rag is None:
        _default_rag = RAGEngine(**kwargs)
    return _default_rag


def query_rag(question: str, top_k: int = 4) -> str:
    """
    Hızlı RAG sorgusu. Doküman yoksa boş string döner.

    Kullanım (ajan promptlarında):
        context = query_rag("THYAO bilanço özeti")
        prompt = f"Bağlam:\n{context}\n\nSoru: ..."
    """
    engine = get_rag_engine()
    return engine.query(question, top_k=top_k)
