import json
import os

import chromadb
import streamlit as st
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from pypdf import PdfReader

load_dotenv()

# ============================================================================
# Agentic RAG サンプル
# ----------------------------------------------------------------------------
# Naive RAG（rag_chat_test）との違い:
#   Naive RAG : 「必ず1回だけ検索 → プロンプトに埋め込み → 生成」という固定の流れ。
#               LLMは検索結果を受け取るだけで、検索の要否や検索語を決められない。
#   Agentic RAG: 検索を「ツール（関数）」としてLLMに渡す。LLM（=エージェント）が
#               ・そもそも検索が必要か
#               ・どんなクエリで検索するか（質問のリライト）
#               ・足りなければ何回でも検索し直すか
#               を自分で判断しながら、最終回答に到達するまでループする。
# ============================================================================

# ===== LLMクライアントの設定 =====
# .envのLLM_PROVIDERでollama / openai / chatai を切り替える
provider = os.getenv("LLM_PROVIDER", "ollama")

if provider == "openai":
    llm_model = os.getenv("OPENAI_MODEL", "gpt-5.4")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif provider == "chatai":
    llm_model = os.getenv("CHATAI_MODEL", "gpt-5.1")
    chatai_base = os.getenv("CHATAI_API_BASE_URL").rstrip("/")
    client = AzureOpenAI(
        api_key=os.getenv("CHATAI_API_KEY"),
        azure_endpoint=f"{chatai_base}/{llm_model}",
        api_version=os.getenv("CHATAI_API_VERSION","DUMMY"),
    )
else:
    # OllamaはOpenAI互換APIを持つのでOpenAIクライアントがそのまま使える
    # 注意: Agentic RAGはツール呼び出し(function calling)を使うため、
    #       Ollama利用時はツール対応モデル（例: qwen3, llama3.1 等）を指定すること
    llm_model = os.getenv("OLLAMA_MODEL", "qwen3.5:9b")
    client = OpenAI(
        api_key="ollama",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    )

# ===== ChromaDB（ベクトルデータベース）の設定 =====
# ベクトルDBはテキストを数値（ベクトル）に変換して保存し、意味的な類似検索ができるDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")


# embeddingモデルの設定（日本語がわかる組み込み用モデルを指定）
# @st.cache_resource でサーバー起動時の1回だけロードし、以降はキャッシュを再利用する
@st.cache_resource
def load_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="cl-nagoya/ruri-v3-30m"
    )


ef = load_embedding_function()

# ドキュメント登録時に自動でembeddingを生成してくれるように、コレクションにembedding_functionを指定して作成
if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.get_or_create_collection(
        name="local_docs", embedding_function=ef
    )


# PDFファイルを読み込む関数
def load_pdf(file):
    # pypdfでPDFを読み込み、全ページのテキストを結合して返す
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ===== テキスト分割 =====
# chunk_size=200
# chunk_overlap=50：前後のチャンクと100文字重複させることで文脈の切れ目をなくす
def split_text(text):
    chunk_size = 200
    overlap = 50
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return chunks


# ===== 検索ツール（エージェントが呼び出す「道具」） =====
# Naive RAGでは検索はコード側が勝手に1回実行していたが、
# Agentic RAGでは検索を関数として用意し、いつ・どんなクエリで呼ぶかをLLMに委ねる。
def search_documents(query: str, n_results: int = 3) -> str:
    """ベクトルDBから質問に関連するドキュメントのチャンクを検索して返す。"""
    count = st.session_state.collection.count()
    if count == 0:
        return "（ドキュメントが登録されていません）"
    n = min(n_results, count)
    results = st.session_state.collection.query(query_texts=[query], n_results=n)
    docs = results["documents"][0] if results["documents"] else []
    if not docs:
        return "（関連するドキュメントは見つかりませんでした）"
    return "\n---\n".join(docs)


# LLMに渡すツール定義（OpenAI互換のtools仕様）
# このスキーマを見て、LLMが「search_documentsをこの引数で呼びたい」と判断する。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "アップロードされたPDFドキュメントの中から、"
                "質問に関連する箇所を意味的類似検索で取得する。"
                "資料に基づいて答える必要があるときに使う。"
                "情報が足りなければクエリを変えて何度でも呼んでよい。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "検索に使う問い合わせ文。質問をそのまま使わず、検索に効くキーワードへ言い換えてもよい。",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "取得するチャンク数（既定3）。",
                    },
                },
                "required": ["query"],
            },
        },
    }
]


# ===== エージェントループ（Agentic RAGの中核） =====
# LLMにツールを渡して呼び出し、ツール呼び出しがあれば実行→結果を返す、を繰り返す。
# ツール呼び出しが無くなった（=最終回答にたどり着いた）時点でループを抜ける。
def run_agent(messages, status_box, max_steps=5):
    # 実行したステップ（検索クエリなど）を後から見返せるように記録していく
    steps = []
    for step in range(max_steps):
        completion = client.chat.completions.create(
            model=llm_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 検索するかどうかはLLMが判断
            temperature=0.3,
        )
        msg = completion.choices[0].message

        # ツール呼び出しが無ければ、それが最終回答
        if not msg.tool_calls:
            return msg.content or "", steps

        # アシスタントのツール呼び出しメッセージを履歴に追加
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # 要求された各ツールを実行し、結果を tool ロールで返す
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc.function.name == "search_documents":
                query = args.get("query", "")
                n_results = int(args.get("n_results", 3))
                line = f"🔍 検索 (step {step + 1}): `{query}`"
                steps.append(line)  # 履歴用に記録
                status_box.write(line)  # 実行中のライブ表示
                result = search_documents(query, n_results)
            else:
                result = f"（未知のツール: {tc.function.name}）"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    # ステップ上限に達した場合は、ツール無しで最終回答を一度だけ生成
    final = client.chat.completions.create(
        model=llm_model, messages=messages, temperature=0.3
    )
    return final.choices[0].message.content or "", steps


PAGE_TITLE = "Agentic RAG Chat Test (PDF)"
st.set_page_config(page_title=PAGE_TITLE)

# Agentic RAGでは、検索の主導権をLLMに渡すための指示をシステムプロンプトに書く
system_prompt = (
    "あなたは資料に基づいて回答するアシスタントです。"
    "ユーザーの質問に答えるために、必要に応じて search_documents ツールを使い、"
    "アップロードされたPDFから根拠となる情報を検索してください。"
    "一度の検索で情報が不足する場合は、クエリを言い換えて検索し直して構いません。"
    "資料に基づく内容は根拠を踏まえて答え、資料に無い内容はその旨を述べてください。"
    "最終的な回答は日本語で行ってください。"
)

# サイドバーに現在使用中のプロバイダ・モデルを表示
st.sidebar.caption(f"LLM: [{provider}] {llm_model}")
st.sidebar.caption("モード: Agentic RAG（検索ツールをLLMが自律的に呼び出し）")

# PDFファイルのアップロード
uploaded_files = st.sidebar.file_uploader(
    "PDFファイルをアップロード", type=["pdf"], accept_multiple_files=True
)

# ===== インデックス作成（RAGの「R」= Retrieval の準備） =====
# アップロード済みファイル名を記録しておき、新しいファイルだけインデックスを作成する
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]

if new_files:
    with st.sidebar.status("インデックス作成中..."):
        for file in new_files:
            chunks = split_text(load_pdf(file))  # PDFからテキスト抽出・分割
            st.session_state.collection.add(
                documents=chunks,  # テキストを登録（embeddingは自動生成）
                ids=[
                    f"{file.name}_{i}" for i in range(len(chunks))
                ],  # 重複しないIDが必要
            )
            st.session_state.indexed_files.add(file.name)
    st.sidebar.success("インデックス作成完了")

# タイトル
st.title(PAGE_TITLE)

# 会話の履歴を保管
if "messages" not in st.session_state:
    st.session_state.messages = []

# 会話の履歴をリセットするボタン
if st.sidebar.button("会話をリセット"):
    st.session_state.messages = []

# 会話の履歴を表示
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        # この回答でエージェントが踏んだ検索ステップを、折りたたみで残しておく
        if m.get("steps"):
            with st.expander(f"🔍 実行ステップ（{len(m['steps'])}件）"):
                for s in m["steps"]:
                    st.write(s)
        st.write(m["content"])


prompt = st.chat_input("メッセージを入力")

if prompt:

    # ユーザーのプロンプトを表示
    with st.chat_message("user"):
        st.write(prompt)

    # 表示用・履歴用にユーザーの発言を保存
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ===== エージェントへ渡すメッセージを構築 =====
    # Naive RAGのように事前検索はせず、検索の判断はエージェントに任せる。
    agent_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]

    # ===== エージェント実行（検索⇄推論のループ）→ 最終回答 =====
    with st.chat_message("assistant"):
        with st.status("エージェントが考え中...", expanded=True) as status:
            answer, steps = run_agent(agent_messages, status)
            # 完了してもステップを畳まず開いたまま残す（あとで動きを確認できるように）
            status.update(label="完了", state="complete", expanded=True)
        st.write(answer)

    # 会話の履歴を保存（ステップも一緒に保存して、次のやり取り後も見返せるようにする）
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "steps": steps}
    )
