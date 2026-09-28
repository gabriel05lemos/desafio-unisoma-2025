# Chatbot Oikon — RAG para conhecimento em Heulosofia

[![Linguagem](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Interface](https://img.shields.io/badge/Interface-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LLM](https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai&logoColor=white)](https://platform.openai.com/)
[![Busca vetorial](https://img.shields.io/badge/Busca-FAISS-0467DF)](https://github.com/facebookresearch/faiss)
[![Status](https://img.shields.io/badge/Status-projeto%20acad%C3%AAmico%20%2F%20MVP-blue)](#status-e-escopo)

## Visão geral

O Chatbot Oikon é uma aplicação de perguntas e respostas baseada em **Retrieval-Augmented Generation (RAG)**. O projeto foi desenvolvido para o Desafio UniSoma 2025, em parceria com a Fundação Eufraten e o Instituto Oikon, para tornar pesquisável o conhecimento presente em palestras e documentos sobre a filosofia Heulosofia.

Em vez de treinar um modelo próprio, a aplicação prepara documentos fornecidos pelo usuário, cria embeddings pela API da OpenAI e armazena os vetores em um índice FAISS. Quando uma pergunta é feita, os trechos semanticamente mais próximos são recuperados do FAISS/SQLite e enviados como contexto para um modelo de chat. A resposta é orientada pelos documentos carregados e pode exibir suas fontes.

A interface é construída em Streamlit e inclui configuração da chave da API, ingestão de TXT/PDF, conversa, monitoramento de tokens/custos e um guia de uso. O armazenamento é local, mas geração, moderação, destilação e embeddings são realizados por chamadas à OpenAI.

## Status e escopo

Este repositório representa um MVP/projeto acadêmico apresentado no Desafio UniSoma 2025. O foco é demonstrar uma aplicação RAG local para documentos da organização; não há pipeline de treinamento supervisionado, serviço de produção, testes automatizados ou benchmark formal de qualidade incluído.

### Resultado e impacto

O projeto foi desenvolvido pela equipe de **Guilherme de Medeiros e Gabriel Lemos (UNICAMP)**. A solução resultou na vitória da equipe no Desafio UniSoma 2025 e foi posteriormente implementada na ONG parceira, levando o chatbot ao contexto real de uso para apoiar a consulta ao acervo de aulas e documentos.

## Principais resultados e métricas

As métricas abaixo são parâmetros e resultados obtidos durante o desafio; não representam benchmark de qualidade do modelo.

| Métrica | Valor | Origem |
| --- | ---: | --- |
| Chunks processados | **1.073** | Arquivos forcenidos pelos organizadores |
| Arquivos de origem históricos | **18** | Arquivos forcenidos pelos organizadores |
| Documentos recuperados por pergunta | **até 5** | `buscar_vetores(..., k=5)` |
| Limite de contexto enviado ao LLM | **9.000 caracteres** | `build_context(..., max_chars=9000)` |
| Tamanho de chunk | **1.500 caracteres** | `split_text_into_chunks` |
| Sobreposição entre chunks | **200 caracteres** | `split_text_into_chunks` |
| Limite de pergunta | **600 caracteres** | Guarda-corpo da aba `Perguntar` |
| Métricas de custo | **USD por chamada** | Tabela `api_calls` e `pricing.py` |

### Métricas apresentadas no material do desafio

O PDF de apresentação do desafio reporta as métricas abaixo como resultados/objetivos da solução. Elas são métricas de impacto apresentadas pela equipe; não foram reproduzidas por um benchmark automatizado neste repositório.

| Métrica reportada | Valor | Origem |
| --- | ---: | --- |
| Economia em relação a modelos recentes | **90% mais barata** | Comparação de custo apresentada pela equipe |
| Fidelidade aos documentos | **98%** | Baseado nos testes durante o desafio |
| Fidelidade contra premissas falsas | **100% projetado** | Meta de comportamento do prompt de analista crítico |
| Capacidade de documentos locais | **mais de 50.000** | Escala indicada para a arquitetura local |
| Capacidade em nuvem | **mais de 500.000** | Escala indicada para uma implantação em nuvem |
| Custo aproximado por pergunta | **R$ 0,06** | Estimativa baseado nos custos de chamadas durante o desafio |
| Custo aproximado de carga de uso | **R$ 3 para cerca de 2.000 chamadas** | Estimativa baseado nos custos de chamadas durante o desafio |

O material também apresenta uma captura do dashboard com aproximadamente **2.148 chamadas** e **US$ 0,64259** de custo total. Esse número é uma fotografia de uma execução demonstrativa, não uma garantia de custo futuro. 
## Metodologia e arquitetura

### Pré-processamento e ingestão

1. O usuário faz upload de arquivos `.txt` ou `.pdf`.
2. `extract_text` lê TXT em UTF-8 e extrai PDF com `pdfplumber`.
3. `is_transcription` identifica transcrições procurando tags no formato `[SPEAKER_n]` nos primeiros 2.000 caracteres.
4. `split_text_into_chunks` usa janela de 1.500 caracteres com sobreposição de 200; o corte é por caracteres, não por tokens ou sentenças.
5. `normalize_text` remove caracteres de largura zero, compacta espaços e remove espaços nas extremidades.
6. Transcrições passam por destilação com `gpt-3.5-turbo`, que remove tags, hesitações e repetições preservando conceitos. Documentos comuns seguem sem reescrita.
7. Cada texto final recebe embedding de `text-embedding-3-small`.
8. Texto e metadados são salvos no SQLite; o vetor é salvo no FAISS usando o `ROWID` do SQLite como identificador.


### Recuperação e geração

Para cada pergunta, o sistema aplica moderação de entrada, rejeita perguntas acima de 600 caracteres, gera o embedding, executa busca L2 no `IndexFlatL2`/`IndexIDMap`, consulta os textos correspondentes no SQLite, monta contexto de até 9.000 caracteres e chama `gpt-3.5-turbo`. O prompt exige resposta baseada exclusivamente no contexto, correção de premissas sem suporte e indicação explícita de lacunas. A resposta, fontes e uso de tokens são registrados.

### Modelagem e avaliação

O sistema não treina um modelo próprio nem escolhe modelos por validação experimental. Usa `text-embedding-3-small` para representação vetorial e `gpt-3.5-turbo` para destilação e resposta. A avaliação existente é operacional (ingestão, recuperação, fontes e custos), não uma avaliação estatística da factualidade.

### Persistência

`utils.py` cria três tabelas SQLite: `documentos` (chunks e arquivos), `api_calls` (timestamps, tipos, tokens e custos) e `configuracao` (chave/valor). O FAISS guarda os vetores e depende dos IDs do SQLite. Esses dados são artefatos locais derivados e não devem ser versionados.

## Stack tecnológica

### Linguagem e aplicação

- Python;
- Streamlit para interface e estado de sessão;
- Pandas para o dashboard;
- SQLite para persistência local.

### IA, busca e documentos

- OpenAI Python SDK para moderação, embeddings e chat completions;
- `text-embedding-3-small` e `gpt-3.5-turbo`;
- FAISS (`faiss-cpu`) com `IndexFlatL2` + `IndexIDMap`;
- `pdfplumber` para PDF;
- NumPy para vetores `float32`.

### Empacotamento

- PyInstaller;
- `ChatbotUnisoma.spec`;
- `.streamlit/config.toml`;
- `requirements.txt`.

## Estrutura do repositório

```text
.
├── .streamlit/
│   └── config.toml          # Tema visual do Streamlit
├── .gitignore               # Ignora ambientes, segredos, dados e builds
├── ChatbotUnisoma.spec      # Receita do PyInstaller
├── Apresentação.pdf         # Documento do desafio (referência)
├── ManualChatbot.pdf        # Manual de uso
├── README.md                # Documentação principal
├── app.py                   # Interface e orquestração RAG
├── pricing.py               # Estimativa de custo por tokens
├── respostas.txt            # Exemplos; não é necessário para runtime
├── requirements.txt         # Dependências Python
├── run_app.py               # Launcher Streamlit/PyInstaller
└── utils.py                 # SQLite, FAISS, embeddings, moderação e chunks
```


## Como executar

### Pré-requisitos

- Python 3.11 ou compatível;
- Git, internet e chave OpenAI válida;
- Windows, macOS ou Linux.

### Clonar e criar ambiente

```bash
git clone https://github.com/gabriel05lemos/desafio-unisoma-2025.git
cd desafio-unisoma-2025
```

Windows PowerShell:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Windows Prompt de Comando:

```bat
py -3.11 -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Instalar e executar

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Também é possível usar o launcher do empacotamento:

```bash
python run_app.py
```

Abra `http://localhost:8501`. Na aba **Configuração**, valide a chave; em **Adicionar Arquivos**, processe TXT/PDF; em **Perguntar**, envie uma questão de até 600 caracteres; em **Monitoramento**, consulte chamadas e custo estimado.

### Compilar executável com PyInstaller

Com o ambiente virtual ativado, execute a especificação versionada:

```bash
python -m PyInstaller --clean --noconfirm ChatbotUnisoma.spec
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python -m PyInstaller --clean --noconfirm ChatbotUnisoma.spec
```

macOS:

```bash
source venv/bin/activate
python -m PyInstaller --clean --noconfirm ChatbotUnisoma.spec
```

Linux:

```bash
source venv/bin/activate
python -m PyInstaller --clean --noconfirm ChatbotUnisoma.spec
```

## Autores e contatos:

`Gabriel Lemos`
- **LinkedIn:** [gabriel5lemos](https://www.linkedin.com/in/gabriel5lemos)
- **E-mail:** [gabriel05lemos@gmail.com](mailto:gabriel05lemos@gmail.com)
- **GitHub:** [gabriel05lemos](https://github.com/gabriel05lemos)


`Guilherme Medeiros`
- **LinkedIn:** [guilhermedemedeiros](https://www.linkedin.com/in/guilhermedemedeiros)
- **E-mail:** [guilherme.ggcm20@gmail.com](mailto:guilherme.ggcm20@gmail.com)
- **GitHub:** [medeirosdev](https://github.com/medeirosdev)
