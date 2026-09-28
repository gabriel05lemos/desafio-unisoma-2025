import sqlite3
import numpy as np
import faiss
from openai import OpenAI
import re
from datetime import datetime

from pricing import calculate_cost

def inicializar_banco(db_path):
    """Cria o banco de dados SQLite e as tabelas se não existirem."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # Tabela para documentos
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documentos (
                id TEXT PRIMARY KEY,
                resumo TEXT,
                filename TEXT
            )
        """)
        # Tabela para monitoramento
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                call_type TEXT,
                model TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                total_tokens INTEGER,
                cost_usd REAL
            )
        """)
        # Tabela para configurações
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS configuracao (
                chave TEXT PRIMARY KEY,
                valor TEXT
            )
        """)
        conn.commit()

# --- Funções para o TOKEN ---
def save_api_token(db_path, token):
    """Salva ou atualiza o token da API no banco de dados."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO configuracao (chave, valor) VALUES (?, ?)", ('openai_api_key', token))
        conn.commit()

def load_api_token(db_path):
    """Carrega o token da API do banco de dados."""
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            result = cursor.execute("SELECT valor FROM configuracao WHERE chave = ?", ('openai_api_key',)).fetchone()
            return result[0] if result else None
    except:
        return None

# --- Função de Log Corrigida ---
def log_api_call(db_path, call_type, model, usage):
    """Registra uma chamada de API no banco de dados de monitoramento."""
    # CORREÇÃO DEFINITIVA: Garante que os valores de tokens sejam sempre números inteiros.
    prompt_tokens = int(usage.prompt_tokens or 0)
    completion_tokens = int(getattr(usage, 'completion_tokens', 0) or 0)
    total_tokens = int(usage.total_tokens or 0)

    print(f"Logging API call: {call_type}, Model: {model}, Prompt Tokens: {prompt_tokens}, Completion Tokens: {completion_tokens}, Total Tokens: {total_tokens}")

    cost = calculate_cost(model, prompt_tokens, completion_tokens)
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_calls (timestamp, call_type, model, prompt_tokens, completion_tokens, total_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(), 
            str(call_type), 
            str(model), 
            int(prompt_tokens), 
            int(completion_tokens), 
            int(total_tokens), 
            float(cost)
        ))
        conn.commit()

def adicionar_vetor(db_path, faiss_index_path, doc_id, resumo, filename, embedding):
    """Adiciona um vetor ao banco de dados e ao índice FAISS."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documentos (id, resumo, filename) VALUES (?, ?, ?)", (doc_id, resumo, filename))
    last_id = cursor.lastrowid
    conn.commit()
    conn.close()

    try:
        index = faiss.read_index(faiss_index_path)
    except RuntimeError:
        dimension = len(embedding)
        index = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIDMap(index)

    index.add_with_ids(np.array([embedding], dtype=np.float32), np.array([last_id]))
    faiss.write_index(index, faiss_index_path)

def buscar_vetores(db_path, faiss_index_path, query_vec, k=5):
    """Busca os vetores mais próximos no FAISS e retorna os documentos do SQLite."""
    try:
        index = faiss.read_index(faiss_index_path)
    except RuntimeError:
        return []

    distances, ids = index.search(np.array([query_vec], dtype=np.float32), k)
    
    if len(ids[0]) == 0 or ids[0][0] < 0:
        return []

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        sql_ids = tuple(int(i) for i in ids[0] if i >= 0)
        if not sql_ids: return []
        
        results = cursor.execute(f"SELECT id, resumo, filename FROM documentos WHERE ROWID IN {sql_ids}").fetchall()
    
    hits = [{
        "id": row[0],
        "document": row[1],
        "metadata": {"filename": row[2]}
    } for row in results]
    
    return hits

def embed_text(client: OpenAI, model: str, text: str):
    """Gera o embedding e retorna tanto o vetor quanto o uso de tokens."""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding, response.usage

def build_context(hits, max_chars=9000):
    used = []
    parts = []
    total = 0
    for h in hits:
        block = f"[DOC] {h['document']}\n[FONTE: {h['metadata'].get('filename', 'N/A')}]\n\n"
        if total + len(block) > max_chars: break
        parts.append(block)
        used.append(h)
        total += len(block)
    return "".join(parts), used

def _serialize_moderation_result(resp):
    r = resp.results[0]
    def to_dict(x):
        return {k: getattr(x, k) for k in dir(x) if not k.startswith("_") and isinstance(getattr(x, k), (bool, float, int, str, dict))}
    return {"flagged": r.flagged, "categories": to_dict(r.categories), "category_scores": to_dict(r.category_scores)}

def check_moderation_in(client: OpenAI, text: str):
    resp = client.moderations.create(input=text)
    data = _serialize_moderation_result(resp)
    if data["flagged"]: return {"allowed": False, "message": "Conteúdo não permitido.", "details": data}
    return {"allowed": True, "details": data}

def check_moderation_out(client: OpenAI, text: str):
    resp = client.moderations.create(input=text)
    data = _serialize_moderation_result(resp)
    if data["flagged"]: return {"allowed": False, "message": "Bloqueado.", "details": data}
    return {"allowed": True, "details": data}

def normalize_text(s: str) -> str:
    s = s or ""
    s = s.replace("\u200b", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Coloque isso no final do seu utils.py

def split_text_into_chunks(text, chunk_size=1500, chunk_overlap=200):
    """Fatia um texto em pedaços (chunks) com sobreposição."""
    if text is None: return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - chunk_overlap)
    return chunks