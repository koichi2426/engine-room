# ベースイメージとして軽量な公式Pythonイメージを使用します
FROM python:3.11-slim-bookworm

# コンテナ内の作業ディレクトリを指定します
WORKDIR /app

# 依存関係ファイルを先にコピーします
COPY requirements.txt .

# pipを更新し、依存関係をインストールします
# キャッシュを使わないことで、イメージサイズを小さく保ちます
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ソースコード（appディレクトリ）をコンテナにコピーします
COPY ./app .

# コンテナの8000番ポートを外部に公開します
EXPOSE 8000

# コンテナ起動時に実行するコマンドです
# uvicornサーバーを起動し、外部からのアクセスを許可(--host 0.0.0.0)し、
# コードの変更を検知して自動リロード(--reload)するようにします
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]