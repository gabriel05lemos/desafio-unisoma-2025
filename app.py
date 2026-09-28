import streamlit as st
import pdfplumber
from openai import OpenAI
import time
import sys 
import os
import uuid
import sqlite3
import pandas as pd
import re 
from pathlib import Path

from utils import (
    inicializar_banco,
    adicionar_vetor,
    buscar_vetores,
    embed_text,
    build_context,
    check_moderation_in,
    check_moderation_out,
    normalize_text,
    log_api_call,
    save_api_token, 
    load_api_token,
    split_text_into_chunks
)

# Define o caminho base para o PyInstaller (simplificado, sem sys._MEIPASS)
if getattr(sys, 'frozen', False):
    # Se estiver rodando como um .exe, o caminho de dados é o diretório do executável
    base_path = os.path.dirname(sys.executable)
else:
    # Se estiver rodando como um script normal
    base_path = os.path.dirname(os.path.abspath(__file__))

# Caminhos para os arquivos de dados (agora sem os caminhos de imagem)
DB_PATH = os.path.join(base_path, "chatbot_data.sqlite3")
FAISS_INDEX_PATH = os.path.join(base_path, "vector_index.faiss")

# --- Lógica do Tema (Necessário o arquivo config.toml) ---
# O arquivo config.toml DEVE existir ao lado do app.py (dentro da pasta .streamlit)
theme_content = """
[theme]
primaryColor="#f37021"
backgroundColor="#3d508a"
secondaryBackgroundColor="#2a3a6a"
textColor="#FFFFFF"
font="sans serif"
"""
config_dir = Path(base_path) / ".streamlit"
config_file = config_dir / "config.toml"

if not config_file.exists():
    config_dir.mkdir(exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(theme_content)

# --- CONFIGURAÇÃO DA PÁGINA (sem page_icon) ---
st.set_page_config(
    page_title="Chatbot Oikon", 
    layout="wide"
)

# Inicializa o banco de dados
inicializar_banco(DB_PATH)

# --- LÓGICA DE CARREGAMENTO E AUTO-VALIDAÇÃO DO TOKEN ---
if "token_initialized" not in st.session_state:
    loaded_token = load_api_token(DB_PATH)
    if loaded_token:
        st.session_state.api_token = loaded_token
        try:
            client = OpenAI(api_key=loaded_token)
            client.moderations.create(input="auto-validate")
            st.session_state.token_validated = True
        except Exception:
            st.session_state.token_validated = False
    else:
        st.session_state.api_token = ""
        st.session_state.token_validated = False
    st.session_state.token_initialized = True
# --- FIM DA LÓGICA ---


# ---- UI DA BARRA LATERAL (SIDEBAR) ----
# Removida st.sidebar.image(LOGO_PATH)
st.sidebar.title("Chatbot Oikon")
st.sidebar.divider()

menu_options = [
    "❓ Perguntar",
    "📤 Adicionar Arquivos",
    "⚙️ Configuração",
    "📊 Monitoramento",
    "📚 Guia de Uso"
]
menu = st.sidebar.radio("Navegação:", menu_options)


# ---- Funções Auxiliares (sem mudança) ----
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text() is not None])
    return ""

def is_transcription(text_sample):
    if not text_sample:
        return False
    if re.search(r'\[SPEAKER_\d+\]', text_sample):
        return True
    return False

# ---- ABA: Configuração ----
if menu == "⚙️ Configuração":
    st.header("Configuração da Chave OpenAI")
    st.write("Insira sua chave da API da OpenAI. Ela será salva localmente de forma segura para futuros acessos.")
    
    api_token_input = st.text_input(
        "Token da API OpenAI", 
        type="password", 
        value=st.session_state.get("api_token", "")
    )
    
    if st.button("Validar e Salvar Token"):
        if not api_token_input.strip():
            st.error("Por favor, insira um token válido.")
        else:
            try:
                with st.spinner("Validando..."):
                    test_client = OpenAI(api_key=api_token_input)
                    test_client.moderations.create(input="validate and save")
                
                save_api_token(DB_PATH, api_token_input)
                st.session_state.api_token = api_token_input
                st.session_state.token_validated = True
                st.success("Token validado e salvo com sucesso!")
                time.sleep(1); st.rerun()
            except Exception as e:
                st.error(f"Erro ao validar token: {str(e)}")
                st.session_state.token_validated = False
    
    st.markdown("---")
    if st.session_state.get("token_validated"):
        st.success("Token configurado e validado.")
    else:
        st.warning("Nenhum token válido está configurado. O chatbot não funcionará.")


# ---- ABA: Adicionar Arquivos ----
elif menu == "📤 Adicionar Arquivos":
    st.header("Adicionar Documentos")
    st.write("Faça upload dos arquivos (.txt ou .pdf). O sistema detectará automaticamente se a limpeza de transcrição é necessária.")
    
    if not st.session_state.get("token_validated"):
        st.warning("Configure e valide o token da API na aba '⚙️ Configuração' antes de adicionar arquivos.")
    else:
        uploaded_files = st.file_uploader("1. Selecione os arquivos", accept_multiple_files=True, type=['txt', 'pdf'])
        st.markdown("---")
        process_btn = st.button("2. Processar e Salvar Arquivos", disabled=(not uploaded_files))
        
        if process_btn:
            client = OpenAI(api_key=st.session_state["api_token"])
            total_chunks_processed = 0
            
            cleaning_prompt_template = """
            Você é um editor de transcrições. Sua tarefa é limpar o texto a seguir, que é uma transcrição de palestra.
            Regras:
            1. Remova todas as tags de palestrante (ex: [SPEAKER_01], [SPEAKER_14], etc.).
            2. Remova palavras de preenchimento, hesitações, repetições e conversas que não sejam sobre o tópico principal (ex: "ah, sim, sim", "eu estou cansada", "ninguém teve?").
            3. Transforme o diálogo em um parágrafo de prosa coeso e limpo.
            4. **Importante:** Mantenha todos os conceitos, nomes próprios e informações essenciais, apenas reescreva-os de forma clara.
            
            Texto da Transcrição:
            "{chunk_text}"
            
            Texto Limpo e Destilado:
            """
            
            with st.spinner("Processando arquivos... Isso pode levar vários minutos."):
                for file in uploaded_files:
                    try:
                        raw_text = extract_text(file)
                        if not raw_text or not raw_text.strip():
                            st.warning(f"Arquivo '{file.name}' está vazio ou não pôde ser lido.")
                            continue

                        apply_distillation = is_transcription(raw_text[:2000]) 
                        if apply_distillation:
                            st.info(f"Arquivo '{file.name}' detectado como transcrição. Aplicando limpeza...")
                        else:
                            st.info(f"Arquivo '{file.name}' detectado como documento padrão. Salvando diretamente.")

                        text_chunks = split_text_into_chunks(raw_text)
                        
                        if not text_chunks:
                            st.warning(f"Não foi possível extrair chunks do arquivo '{file.name}'.")
                            continue

                        for i, chunk in enumerate(text_chunks):
                            raw_chunk_text = normalize_text(chunk)
                            if not raw_chunk_text:
                                continue

                            clean_text = ""
                            
                            if apply_distillation:
                                try:
                                    completion = client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "Você é um editor de transcrições focado em limpar e preservar conceitos."},
                                            {"role": "user", "content": cleaning_prompt_template.format(chunk_text=raw_chunk_text)}
                                        ]
                                    )
                                    log_api_call(DB_PATH, 'chat_distill', 'gpt-3.5-turbo', completion.usage)
                                    clean_text = completion.choices[0].message.content
                                
                                except Exception as e:
                                    st.warning(f"Falha ao destilar chunk {i+1}/{len(text_chunks)}: {e}")
                                    continue
                            else:
                                clean_text = raw_chunk_text

                            embedding, usage_embed = embed_text(client, model="text-embedding-3-small", text=clean_text)
                            log_api_call(DB_PATH, 'embedding', 'text-embedding-3-small', usage_embed)
                            
                            doc_id = str(uuid.uuid4())
                            adicionar_vetor(DB_PATH, FAISS_INDEX_PATH, doc_id, clean_text, file.name, embedding)
                            total_chunks_processed += 1
                        
                        st.success(f"Arquivo '{file.name}' processado e salvo!")

                    except Exception as e:
                        st.error(f"Falha ao processar '{file.name}': {e}")
            
            st.success(f"Processamento concluído! {total_chunks_processed} chunks de texto salvos no total.")


# ---- ABA: Perguntar (Sem Avatares de Imagem) ----
elif menu == "❓ Perguntar":
    st.header("Perguntar ao Assistente Oikon")
    
    if not st.session_state.get("token_validated"):
        st.warning("Configure e valide o token da API na aba '⚙️ Configuração' para usar o chat.")
    else:
        
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Olá! Sou o assistente Oikon. Estou pronto para responder suas perguntas com base nos documentos.", "sources": []}
            ]

        # Exibe o histórico de chat
        for message in st.session_state.messages:
            # Avatar será o padrão do Streamlit (emoji ou ícone)
            avatar_img = "assistant" if message["role"] == "assistant" else "user"
            with st.chat_message(message["role"], avatar=avatar_img): 
                st.markdown(message["content"])
                
                if message.get("sources"):
                    with st.expander("Fontes consultadas", expanded=False):
                        if not message["sources"]:
                            st.caption("Nenhuma fonte específica utilizada.")
                        for f in message["sources"]:
                            st.caption(f)

        question = st.chat_input("Pergunte ao assistente Oikon sobre a Heulosofia...")
        
        if question:

            client = OpenAI(api_key=st.session_state["api_token"])

            # --- TRAVA 1: MODERAÇÃO DE CONTEÚDO (GUARDA-CORPO) ---
            moderation_result = check_moderation_in(client, question)
            if not moderation_result["allowed"]:
                st.error("Sua pergunta foi bloqueada por conter conteúdo inadequado. Por favor, reformule a questão de forma respeitosa.")
                # st.stop() para interromper a execução e evitar o processamento
                st.stop() 
            # --- FIM DA MODERAÇÃO ---

            # --- Trava de Segurança Contra Prompts Longos ---
            MAX_QUESTION_LENGTH = 600 # Limite de caracteres para evitar prompts abusivos e custos inesperados
            
            if len(question) > MAX_QUESTION_LENGTH:
                st.error(f"Sua pergunta tem {len(question)} caracteres. Por favor, limite a pergunta a {MAX_QUESTION_LENGTH} caracteres para garantir a estabilidade e o baixo custo do sistema.")
                
                # Usamos st.stop() para encerrar a execução do script para esta sessão, 
                # pois não queremos processar a pergunta.
                st.stop()


            
            
            user_message = {"role": "user", "content": question, "sources": []}
            st.session_state.messages.append(user_message)
            with st.chat_message("user"):
                st.markdown(question)
            
            # Bloco de geração de nova resposta
            with st.chat_message("assistant", avatar="assistant"): 
                
                with st.spinner("Analisando os documentos..."):
                    
                    resposta = ""
                    sources = []
                    
                    try:
                        query_vec, usage_embed_q = embed_text(client, model="text-embedding-3-small", text=question)
                        log_api_call(DB_PATH, 'embedding_query', 'text-embedding-3-small', usage_embed_q)
                        
                        hits = buscar_vetores(DB_PATH, FAISS_INDEX_PATH, query_vec, k=5)
                        
                        if not hits:
                            resposta = "Não encontrei informação sobre isso nos documentos fornecidos."
                            sources = []
                            st.markdown(resposta)
                        else:
                            context, used_hits = build_context(hits)
                            
                            system_prompt = """
                            Você é um assistente especialista e analista crítico da filosofia Heulosofia, associado ao Instituto Oikon. 
                            Sua tarefa é responder perguntas com base **exclusivamente** no contexto fornecido.

                            Siga estas regras rigorosamente:
                            1.  **Analise a Pergunta:** Antes de responder, analise a pergunta do usuário em busca de suposições ou premissas.
                            2.  **Verifique as Premissas:** Compare as premissas da pergunta com os fatos no contexto.
                            3.  **Corrija Premissas Falsas:** Se a pergunta contiver uma premissa que não é suportada ou é contradita pelo contexto (ex: "Por que a Heulosofia acredita que o céu é verde?"), você deve **primeiro corrigir a premissa** (ex: "Com base nos documentos, não há menção de que a Heulosofia acredite que o céu é verde.") antes de prosseguir.
                            4.  **Responda a Tudo:** Responda a **todas as partes** da pergunta do usuário.
                            5.  **Seja Cauteloso:** Se a resposta para qualquer parte da pergunta não estiver explicitamente no contexto, afirme claramente: "Com base nos documentos que analisei, não encontrei uma resposta para [tópico específico]."
                            """
                            
                            completion = client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": f"Contexto:\n{context}\n\nPergunta: {question}"}
                                ]
                            )
                            log_api_call(DB_PATH, 'chat_answer', 'gpt-3.5-turbo', completion.usage)
                            resposta = completion.choices[0].message.content
                            
                            sources = sorted(list(set(h['metadata'].get('filename', 'N/A') for h in used_hits)))
                            
                            st.markdown(resposta)
                            
                            with st.expander("Fontes consultadas", expanded=False):
                                if not sources:
                                    st.caption("Nenhuma fonte específica utilizada.")
                                for f in sources:
                                    st.caption(f)
                    
                    except Exception as e:
                        resposta = f"Desculpe, ocorreu um erro inesperado: {str(e)}"
                        sources = []
                        st.error(resposta)

                assistant_message = {"role": "assistant", "content": resposta, "sources": sources}
                st.session_state.messages.append(assistant_message)

# ---- ABA: Monitoramento ----
elif menu == "📊 Monitoramento":
    st.header("Monitoramento de Custo e Uso")
    
    try:
        if not os.path.exists(DB_PATH):
            st.warning("Nenhum dado de monitoramento encontrado. Faça algumas chamadas à API primeiro.")
        else:
            with sqlite3.connect(DB_PATH) as conn:
                if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_calls'").fetchone():
                    st.info("Nenhuma chamada à API foi registrada ainda.")
                else:
                    df = pd.read_sql_query("SELECT * FROM api_calls", conn)
            
                    if df.empty:
                        st.info("Nenhuma chamada à API foi registrada ainda.")
                    else:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        today = pd.to_datetime('today').normalize()
                        start_of_month = today.replace(day=1)

                        df_today = df[df['timestamp'] >= today]
                        df_month = df[df['timestamp'] >= start_of_month]

                        st.subheader("Resumo do Consumo")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Custo Hoje (USD)", f"${df_today['cost_usd'].sum():.6f}")
                            st.metric("Chamadas Hoje", f"{len(df_today)}")
                        with col2:
                            st.metric("Custo no Mês (USD)", f"${df_month['cost_usd'].sum():.5f}")
                            st.metric("Chamadas no Mês", f"{len(df_month)}")
                        with col3:
                            st.metric("Custo Total (USD)", f"${df['cost_usd'].sum():.5f}")
                            st.metric("Chamadas Totais", f"{len(df)}")

                        st.subheader("Log de Chamadas")
                        st.dataframe(df.sort_values(by="timestamp", ascending=False))
                        
                        st.markdown("---")
                        if st.button("Limpar Logs de Monitoramento"):
                            with sqlite3.connect(DB_PATH) as conn:
                                conn.execute("DELETE FROM api_calls")
                                conn.commit()
                            st.success("Todos os logs de monitoramento foram apagados.")
                            time.sleep(1); st.rerun()

    except Exception as e:
        st.error(f"Não foi possível carregar os dados de monitoramento: {e}")


# ---- ABA: Dúvidas Frequentes ----
elif menu == "📚 Guia de Uso":
    st.header("Guia de Uso Rápido e Perguntas Comuns")
    st.write("Bem-vindo ao guia do Assistente Oikon! Este sistema foi feito para ser simples.")

    with st.expander("Como usar o Assistente? (Passo a Passo)", expanded=True):
        st.markdown("""
            O uso tem apenas três etapas:
            1.  **Chave Secreta:** Primeiro, vá em **'Configuração'** e digite a Chave de Acesso que a equipe forneceu. Clique em "Validar e Salvar Token".
            2.  **Aprender Documentos:** Depois, vá em **'Adicionar Arquivos'**. Clique para selecionar as novas palestras e transcrições da ONG e clique em "Processar e Salvar Arquivos". O assistente vai ler e aprender com o novo material.
            3.  **Perguntar:** Volte para **'Perguntar'** e digite o que você precisa saber sobre a Heulosofia.
        """)

    with st.expander("O que acontece quando eu adiciono um novo arquivo?", expanded=False):
        st.markdown("""
            Quando você adiciona um arquivo, o sistema o prepara para a busca inteligente:
            * **Limpeza Automática:** Se o arquivo for uma transcrição de fala (com 'Palestrante 1:' ou ruídos), o sistema de Inteligência Artificial **limpa o texto** automaticamente, removendo o que não importa (como hesitações e falas curtas) e deixando apenas o conhecimento puro.
            * **Memória do Conhecimento:** O texto limpo é guardado na memória permanente do Assistente, então ele pode ser encontrado facilmente mais tarde.
        """)
        
    with st.expander("Preciso colocar a Chave Secreta toda vez que abro o programa?", expanded=False):
        st.markdown("""
            **Não.** Depois de salvar a chave uma vez na aba **'Configuração'**, ela fica guardada com segurança no seu computador. O Assistente a usará automaticamente nas próximas vezes.
        """)
        
    with st.expander("O Assistente me disse 'Não sei'. Por quê?", expanded=False):
        st.markdown("""
            O Assistente Oikon foi programado para ser **100% fiel** ao conhecimento da ONG.
            
            Se ele não souber a resposta, significa que a informação que você procurou **não foi encontrada nos arquivos** que você carregou. Ele nunca vai "inventar" uma resposta da Internet.
        """)
    
    with st.expander("O conhecimento que carrego é salvo permanentemente?", expanded=False):
        st.markdown("""
            **Sim.** Todos os arquivos que você processa são salvos permanentemente no seu computador (na mesma pasta do programa). Você não precisa recarregá-los.
        """)