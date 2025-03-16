import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import base64
import math
import os
import asyncio
import aiohttp
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import google.generativeai as genai

# 環境変数の読み込み
load_dotenv()

# Google Gemini API設定
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_download_link(df_chunks):
    """Generate a download link for a zip file containing all chunks"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for i, chunk in enumerate(df_chunks):
            csv_buffer = io.StringIO()
            chunk.to_csv(csv_buffer, index=False, header=True)
            zip_file.writestr(f"chunk_{i+1}.csv", csv_buffer.getvalue())
    
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:application/zip;base64,{b64}" download="csv_chunks.zip">ダウンロード: 分割されたCSVファイル</a>'

def calculate_chunks_needed(total_rows, chunk_size=20):
    """指定した行数と分割サイズから必要なチャンク数を計算"""
    return math.ceil(total_rows / chunk_size)

def split_csv_into_chunks(df, chunk_size=20):
    """Split a DataFrame into chunks of specified row size"""
    total_rows = len(df)
    chunks = []
    
    for i in range(0, total_rows, chunk_size):
        end_idx = min(i + chunk_size, total_rows)
        chunk = df.iloc[i:end_idx, :].copy()
        chunks.append(chunk)
    
    return chunks

async def call_openai_api(session, chunk_df, task_type, api_key):
    """非同期でOpenAI APIを呼び出す"""
    # CSVデータを文字列に変換
    csv_content = chunk_df.to_csv(index=False)
    columns_list = chunk_df.columns.tolist()
    
    # タスクに応じたプロンプトを生成
    if task_type == "sales_analysis":
        system_prompt = f"""
        あなたは与えられたCSVデータから販売分析レポートを作成する専門家です。
        データを分析して、以下の内容を含む詳細な分析レポートを作成してください：
        1. 全体的な売上傾向
        2. 主要な指標の概要
        3. データから得られる重要な洞察
        4. 推奨事項（もしあれば）
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、レポート内のテーブルには以下の3つの列のみを表示してください：
        1. 商品名（製品名や品目名などの商品を識別する列）
        2. 型番（製品コードや商品IDなどの識別子）
        3. 値段（価格や売上金額など）
        
        テーブル表示は上記3つの列のみに限定し、他の列のデータは文章での分析に使用してください。
        表示する列が見つからない場合は、最も近いと思われる列を選択してください。
        
        CSVデータには必要な情報が含まれています。返答はマークダウン形式で、
        見出しや箇条書きを使用して読みやすくしてください。
        """
    elif task_type == "ad_analysis":
        system_prompt = f"""
        あなたは与えられたCSVデータから広告パフォーマンス分析レポートを作成する専門家です。
        データを分析して、以下の内容を含む詳細な分析レポートを作成してください：
        1. 広告キャンペーンの全体的なパフォーマンス
        2. ROI、ROAS、CPAなどの主要指標の分析
        3. 最も効果的な広告チャネルや広告タイプの特定
        4. コンバージョン率の分析
        5. 費用対効果の高い広告施策の提案
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、レポート内のテーブルには以下の3つの列のみを表示してください：
        1. 広告名/チャネル名（広告やキャンペーンを識別する列）
        2. 広告ID/コード（キャンペーンの識別子）
        3. パフォーマンス指標（コスト、ROI、コンバージョン率など）
        
        テーブル表示は上記3つの列のみに限定し、他の列のデータは文章での分析に使用してください。
        表示する列が見つからない場合は、データの内容から最も近いと思われる列を選択してください。
        
        CSVデータには必要な情報が含まれています。返答はマークダウン形式で、
        見出しや箇条書きを使用して読みやすくしてください。
        効果的な広告戦略のための具体的な改善提案も含めてください。
        """
    elif task_type == "review_analysis":
        system_prompt = f"""
        あなたは与えられたCSVデータから顧客レビュー・感想分析レポートを作成する専門家です。
        データを分析して、以下の内容を含む詳細な分析レポートを作成してください：
        1. 全体的な顧客満足度と感情分析（ポジティブ/ネガティブの割合）
        2. 最も頻繁に言及されているキーワードやトピックの特定
        3. 製品/サービスの強みと改善が必要な点の分析
        4. 競合他社や類似製品との比較（データに含まれている場合）
        5. 顧客ロイヤリティと再購入意向の分析
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、レポート内のテーブルには以下の3つの列のみを表示してください：
        1. 製品/サービス名（レビュー対象を識別する列）
        2. 評価/スコア（星評価や数値評価などの指標）
        3. キーワード/感想（主要なフィードバックポイントや感想の要約）
        
        テーブル表示は上記3つの列のみに限定し、他の列のデータは文章での分析に使用してください。
        表示する列が見つからない場合は、データの内容から最も近いと思われる列を選択してください。
        
        CSVデータには必要な情報が含まれています。返答はマークダウン形式で、
        見出しや箇条書きを使用して読みやすくしてください。
        顧客満足度を向上させるための具体的な改善提案も含めてください。
        """
    else:
        system_prompt = f"""
        与えられたCSVデータを分析し、有用な洞察を提供してください。
        
        重要: 分析では全ての列({len(columns_list)}列)を考慮してください。
        ただし、結果のテーブルには以下の3つの列のみを表示してください：
        1. 商品名（製品名や品目名などの商品を識別する列）
        2. 型番（製品コードや商品IDなどの識別子）
        3. 値段（価格や売上金額など）
        """
    
    user_prompt = f"""以下のCSVデータを分析してください。
    
分析には全ての列({len(columns_list)}列)を使用してください。
ただし、テーブル表示は「商品名」「型番」「値段」の3つの列のみに限定してください。
これらの列が正確に一致しない場合は、データの内容から最も近いと思われる列を選んでください。

{csv_content}"""
    
    # APIリクエストを準備
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 10000
    }
    
    # 非同期でAPIリクエストを送信
    try:
        async with session.post("https://api.openai.com/v1/chat/completions", 
                               headers=headers, 
                               json=payload) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                return f"APIエラー (ステータスコード: {response.status}): {error_text}"
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

async def process_chunks_with_openai(chunks, task_type, api_key):
    """複数のチャンクを非同期で処理"""
    async with aiohttp.ClientSession() as session:
        tasks = [call_openai_api(session, chunk, task_type, api_key) for chunk in chunks]
        return await asyncio.gather(*tasks)

def run_openai_analysis(chunks, task_type, api_key):
    """OpenAI分析を実行するためのメインエントリポイント"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    results = loop.run_until_complete(process_chunks_with_openai(chunks, task_type, api_key))
    return results

def generate_summary_with_gemini(openai_results, original_df_info, analysis_type, file_name=None):
    """Gemini Flash Thinkingを使用して、OpenAIの分析結果を総括する"""
    try:
        if not GEMINI_API_KEY:
            return "Gemini APIキーが設定されていないため、総括分析を実行できません。「APIキーの設定について」セクションでGemini APIキーを設定してください。"
        
        # Geminiモデルの設定
        model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
        
        # 元データの情報
        rows, cols = original_df_info
        
        # 分析タイプに応じたプロンプト生成
        if analysis_type == "広告分析レポート":
            system_prompt = f"""
            あなたはデータ分析の専門家です。OpenAIによって生成された複数の広告パフォーマンス分析レポートを総括して、
            包括的な最終分析レポートを作成してください。

            以下のポイントに注目してください：
            1. 複数のチャンクの分析結果を統合し、広告キャンペーン全体のパフォーマンスを把握する
            2. 最も効果的な広告チャネルやキャンペーンを特定する
            3. ROI、ROAS、CPAなどの主要指標の全体的な傾向を分析する
            4. コンバージョン率やクリック率の分析
            5. 広告予算の最適な配分方法に関する提案
            6. 改善が必要な広告キャンペーンの特定

            元データは全部で{rows}行×{cols}列のCSVファイルです。
            これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
            
            最終レポートには以下のセクションを含めてください：
            - エグゼクティブサマリー
            - 主要なパフォーマンス指標の分析
            - チャネル別・キャンペーン別の詳細分析
            - ベストプラクティスと改善点
            - 今後の広告戦略に関する推奨事項
            
            テーブルには広告名/チャネル名、広告ID/コード、パフォーマンス指標の情報を明確に示してください。
            データに基づいた具体的な改善策を提案し、広告予算の最適な配分方法についても言及してください。
            """
        elif analysis_type == "レビュー分析レポート":
            system_prompt = f"""
            あなたはデータ分析の専門家です。OpenAIによって生成された複数の顧客レビュー・感想分析レポートを総括して、
            包括的な最終分析レポートを作成してください。

            以下のポイントに注目してください：
            1. 複数のチャンクの分析結果を統合し、全体的な顧客満足度と感情分析を把握する
            2. 最も頻繁に言及されているキーワードやトピックを特定する
            3. 製品/サービスの強みと改善が必要な点を明確にする
            4. 競合他社や類似製品との比較分析（データに含まれている場合）
            5. 顧客セグメント別の傾向分析
            6. 時系列での顧客フィードバックの変化（データに含まれている場合）

            元データは全部で{rows}行×{cols}列のCSVファイルです。
            これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
            
            最終レポートには以下のセクションを含めてください：
            - エグゼクティブサマリー
            - 感情分析の概要（ポジティブ/ネガティブ/中立の割合）
            - 主要なキーワードとトピックの分析
            - 製品/サービスの強みと改善点
            - 顧客満足度向上のための具体的な推奨事項
            - 競合分析（データに含まれている場合）
            
            テーブルには製品/サービス名、評価/スコア、キーワード/感想の情報を明確に示してください。
            顧客体験を向上させるための実用的かつ具体的な改善策を提案してください。
            """
        else:  # 販売分析レポート
            system_prompt = f"""
            あなたはデータ分析の専門家です。OpenAIによって生成された複数の販売分析レポートを総括して、
            包括的な最終分析レポートを作成してください。

            以下のポイントに注目してください：
            1. 複数のチャンクの分析結果を統合し、全体的な売上傾向を把握する
            2. 主要な販売指標の傾向を分析する
            3. 特に注目すべき洞察を提供する
            4. データに基づいた具体的な推奨事項を提案する

            元データは全部で{rows}行×{cols}列のCSVファイルです。
            これは複数のチャンクに分割され、各チャンクごとに分析が行われています。
            
            最終レポートには以下のセクションを含めてください：
            - エグゼクティブサマリー
            - 重要な発見
            - 詳細分析（必要に応じてテーブルを含める）
            - 推奨事項
            - 結論
            
            テーブルには商品名・型番・値段の情報を明確に示してください。
            """
        
        # 各チャンクの分析結果を結合
        combined_analyses = ""
        for i, result in enumerate(openai_results):
            combined_analyses += f"\n---- チャンク{i+1}の分析結果 ----\n{result}\n"
        
        # Geminiへのプロンプト
        prompt = f"""
        {system_prompt}

        以下はOpenAIによる各チャンクの分析結果です：
        {combined_analyses}

        これらの分析結果を統合して、包括的な最終分析レポートを作成してください。
        見やすいマークダウン形式で、適切な見出し、箇条書き、テーブルを使用してください。
        """
        
        # Geminiで総括分析を実行
        response = model.generate_content(prompt)
        summary_text = response.text
        
        # Make.com Webhookにデータを送信
        if enable_webhook and webhook_url:
            try:
                # 送信データの作成
                webhook_data = {
                    "analysis_type": analysis_type,
                    "summary": summary_text,
                    "data_info": {
                        "total_rows": rows,
                        "total_columns": cols,
                        "chunks_analyzed": len(openai_results),
                        "file_name": file_name if file_name else "unknown_file"
                    },
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_metadata": {
                        "model": "gemini-2.0-flash-thinking-exp-01-21",
                        "openai_model": "gpt-4o-mini",
                        "chunks_size": chunk_size
                    }
                }
                
                # セッションステートに送信データを保存（確認用）
                st.session_state.webhook_data = webhook_data
                
                # Webhookへデータ送信
                webhook_response = requests.post(webhook_url, json=webhook_data)
                
                # セッション状態にステータスを保存
                if webhook_response.status_code == 200:
                    st.session_state.webhook_status = f"✅ 分析結果がMake.comに正常に送信されました（ステータス: {webhook_response.status_code}）"
                else:
                    st.session_state.webhook_status = f"❌ Make.comへの送信が失敗しました: {webhook_response.status_code}"
            except Exception as e:
                st.session_state.webhook_status = f"❌ Webhookへの送信中にエラーが発生しました: {str(e)}"
        
        # 結果を返す
        return summary_text
    
    except Exception as e:
        return f"Geminiによる総括分析中にエラーが発生しました: {str(e)}"

# Streamlitアプリの初期化
if 'webhook_status' not in st.session_state:
    st.session_state.webhook_status = ""

st.title("CSVファイル分割ツール & LLM分析")
st.write("CSVファイルを20行ごとのチャンクに分割し、OpenAIによる分析を実行します")

# サイドバー設定
st.sidebar.write("## 設定")
chunk_size = 20
custom_chunk_size = st.sidebar.number_input("チャンクサイズ（行数）", 
                                          min_value=1, 
                                          max_value=100, 
                                          value=chunk_size,
                                          step=1)
if custom_chunk_size != chunk_size:
    chunk_size = custom_chunk_size
    st.sidebar.success(f"チャンクサイズを {chunk_size} 行に設定しました")

# Webhook設定
st.sidebar.write("## Webhook設定")
enable_webhook = st.sidebar.checkbox("Make.com Webhookに分析結果を送信する", value=True)
webhook_url = st.sidebar.text_input(
    "Webhook URL", 
    value="https://hook.us1.make.com/yv19q1vynuxr36v9i663ygk3e2eqv4it",
    type="default" if st.sidebar.checkbox("URLを表示", value=False) else "password"
)
show_webhook_data = st.sidebar.checkbox("送信データを表示", value=False)

if show_webhook_data and 'webhook_data' in st.session_state:
    with st.sidebar.expander("送信データの詳細", expanded=True):
        st.json(st.session_state.webhook_data)

# OpenAI API設定
st.sidebar.write("## API設定")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input("OpenAI APIキー", type="password")
    if api_key:
        st.sidebar.success("APIキーが設定されました")

# ファイルアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        total_rows = len(df)
        st.write(f"アップロードされたCSVファイル: {uploaded_file.name}")
        st.write(f"元のサイズ: {df.shape[0]}行 × {df.shape[1]}列")
        
        chunks_needed = calculate_chunks_needed(total_rows, chunk_size)
        st.write(f"分割予測: {total_rows}行を{chunk_size}行ずつ分割すると、{chunks_needed}個のチャンクになります")
        
        example_chunks = calculate_chunks_needed(700, chunk_size)
        st.write(f"参考: 700行のCSVファイルは{chunk_size}行ずつ分割すると、{example_chunks}個のチャンクになります")
        
        # データサンプル表示
        st.write("データのサンプル:")
        st.dataframe(df.head())
        
        # チャンク分割
        chunks = split_csv_into_chunks(df, chunk_size)
        st.write(f"分割結果: {len(chunks)}チャンク (各チャンクには全ての列と対応する行が含まれます)")
        
        for i, chunk in enumerate(chunks):
            st.write(f"チャンク {i+1}: {chunk.shape[0]}行 × {chunk.shape[1]}列")
        
        # チャンクプレビュー
        chunk_to_view = st.selectbox("プレビューするチャンクを選択:", 
                                   range(1, len(chunks) + 1),
                                   format_func=lambda x: f"チャンク {x}")
        if chunk_to_view:
            st.write(f"チャンク {chunk_to_view} のプレビュー:")
            st.dataframe(chunks[chunk_to_view-1].head())
        
        # CSVダウンロードリンク
        st.markdown(get_download_link(chunks), unsafe_allow_html=True)
        
        # LLM分析セクション
        st.write("---")
        st.write("## OpenAI LLM分析")
        st.write("分割したCSVチャンクを使用してLLM分析を実行できます。テーブルには商品名、型番、値段の3つの列のみが表示されます。")
        
        # 分析タイプ選択
        analysis_type = st.radio(
            "分析タイプを選択してください:",
            ["販売分析レポート", "広告分析レポート", "レビュー分析レポート"]
        )
        
        # チャンク数制限（処理負荷とコスト削減のため）
        max_chunks_for_analysis = min(5, len(chunks))
        chunks_for_analysis = st.slider(
            "分析に使用するチャンク数を選択 (コストと処理時間を考慮して制限してください):",
            1, len(chunks), min(3, len(chunks))
        )
        
        if st.button("LLM分析を実行"):
            if not api_key:
                st.error("OpenAI APIキーが設定されていません。サイドバーでAPIキーを入力してください。")
            else:
                if analysis_type == "広告分析レポート":
                    task_type = "ad_analysis"
                elif analysis_type == "レビュー分析レポート":
                    task_type = "review_analysis"
                else:
                    task_type = "sales_analysis"
                
                # プログレスバーの表示
                progress = st.progress(0)
                status_text = st.empty()
                
                status_text.text("分析の準備中...")
                time.sleep(1)
                
                # 分析用のチャンクを準備（選択された数だけ）
                selected_chunks = chunks[:chunks_for_analysis]
                
                # 分析開始
                status_text.text("LLMによる分析を実行中...")
                
                try:
                    # 並列処理で分析を実行
                    results = run_openai_analysis(selected_chunks, task_type, api_key)
                    
                    # 結果表示
                    status_text.text("分析完了！結果を表示します")
                    progress.progress(100)
                    
                    # タブでチャンクごとの分析結果を表示
                    st.write(f"### {analysis_type}の結果")
                    tabs = st.tabs([f"チャンク {i+1}" for i in range(len(selected_chunks))])
                    
                    for i, (tab, result) in enumerate(zip(tabs, results)):
                        with tab:
                            st.markdown(result)
                    
                    # すべての結果を結合した総合分析も表示
                    with st.expander("すべてのチャンクの分析結果一覧"):
                        combined_results = "\n\n".join([f"**チャンク {i+1}の分析**\n{res}" for i, res in enumerate(results)])
                        st.markdown(combined_results)
                    
                    # Geminiによる総括分析の実行
                    st.write("---")
                    st.write("## Geminiによる総括分析")
                    
                    with st.spinner("Gemini AIによる総括分析を実行中..."):
                        # 元データの情報
                        original_df_info = (df.shape[0], df.shape[1])
                        
                        # Webhookステータス表示用
                        webhook_status = st.empty()
                        if enable_webhook:
                            webhook_status.info("分析結果をMake.com Webhookに送信する準備ができています...")
                        
                        # ファイル名を取得
                        file_name = uploaded_file.name if uploaded_file else None
                        
                        # Geminiによる総括分析を実行
                        summary = generate_summary_with_gemini(results, original_df_info, analysis_type, file_name)
                        
                        # Webhook送信結果の表示
                        if enable_webhook:
                            if "正常に送信" in st.session_state.get('webhook_status', ''):
                                webhook_status.success(st.session_state.webhook_status)
                            elif "失敗" in st.session_state.get('webhook_status', ''):
                                webhook_status.error(st.session_state.webhook_status)
                            else:
                                webhook_status.empty()
                        
                        # 結果を表示
                        st.markdown(summary)
                    
                except Exception as e:
                    st.error(f"分析中にエラーが発生しました: {str(e)}")
                    
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

# .envファイルがない場合は作成するオプション
st.write("---")
with st.expander("APIキーの設定について"):
    st.write("""
    1. OpenAI APIキーとGemini APIキーをサイドバーに直接入力するか、
    2. プロジェクトのルートディレクトリに`.env`ファイルを作成し、以下の形式で保存することができます:
    ```
    OPENAI_API_KEY=your-openai-api-key-here
    GEMINI_API_KEY=your-gemini-api-key-here
    ```
    セキュリティのため、`.env`ファイルを使用することをお勧めします。
    """)
    
    if st.button(".envファイルを作成"):
        if api_key or gemini_api_key:
            try:
                env_content = ""
                if api_key:
                    env_content += f"OPENAI_API_KEY={api_key}\n"
                if gemini_api_key:
                    env_content += f"GEMINI_API_KEY={gemini_api_key}\n"
                
                with open(".env", "w") as f:
                    f.write(env_content)
                st.success(".envファイルが作成されました。")
            except Exception as e:
                st.error(f".envファイルの作成に失敗しました: {e}")
        else:
            st.warning("APIキーが入力されていません。サイドバーでAPIキーを入力してから再度お試しください。") 