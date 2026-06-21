# Agentic RAG Chat Test (PDF)

PDF をアップロードして、その内容をもとに質問できる **Agentic RAG** チャットアプリです。
`rag_chat_test`（Naive RAG）を改造したサンプルです。

## Naive RAG との違い

| | Naive RAG (`rag_chat_test`) | Agentic RAG (`agentic_rag_chat_test`) |
|---|---|---|
| 検索の実行 | 質問ごとに**必ず1回**コード側が実行 | **LLM（エージェント）が必要に応じて**実行 |
| 検索クエリ | ユーザーの質問そのまま | LLM が検索向けに**言い換え可能** |
| 検索回数 | 1回固定 | 情報が足りなければ**何度でも**検索し直す |
| 仕組み | retrieval → augment → generate の固定パイプライン | 検索を**ツール（function calling）**として渡し、推論⇄検索をループ |

検索は `search_documents` ツールとして LLM に渡され、「いつ・どんなクエリで・何件取るか」を
LLM 自身が判断します。最終回答にたどり着くまで検索と推論を繰り返します。

## 機能

- PDF のアップロードとベクトルインデックス作成
- ChromaDB による意味的類似検索（RAG）
- LLM による自律的な検索ツール呼び出し（Agentic RAG）
- 検索ステップ（実行クエリ）の可視化
- Ollama / OpenAI の切り替え対応

## セットアップ

```cmd
# 依存パッケージのインストール
uv sync

# 環境変数の設定
copy .env.example .env
# .env を編集してプロバイダ・APIキーを設定
```

## 起動

```cmd
uv run streamlit run agentic_rag_chat_test.py --server.fileWatcherType none
```

`sentence-transformers` + `transformers` の組み合わせでは、Streamlit のファイル監視が `torchvision` の未導入モジュールを走査して大量の `ModuleNotFoundError` ログを出すことがあります。`--server.fileWatcherType none` で監視を無効化すると回避できます。

## 注意（Ollama 利用時）

Agentic RAG はツール呼び出し（function calling）を使います。Ollama を使う場合は
**ツール対応モデル**（例: `qwen3`, `llama3.1` など）を `OLLAMA_MODEL` に指定してください。
ツール非対応モデルでは検索が呼び出されません。

## 環境変数

| 変数名 | 説明 | デフォルト |
|---|---|---|
| `LLM_PROVIDER` | 使用するプロバイダ (`ollama` or `openai`) | `ollama` |
| `OLLAMA_BASE_URL` | Ollama の API エンドポイント | `http://localhost:11434/v1` |
| `OLLAMA_MODEL` | Ollama で使用するモデル名（ツール対応モデル推奨） | `qwen3.5:9b` |
| `OPENAI_API_KEY` | OpenAI の API キー | — |
| `OPENAI_MODEL` | OpenAI で使用するモデル名 | `gpt-5.1` |
