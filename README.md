# Gemini 3 Image 画像生成CLI（REST版）

`Gemini API` に画像生成プロンプトを投げて、返ってきた `inlineData.data`（base64）を **画像ファイルとしてローカルに保存**するための、シンプルなPythonスクリプトです。

このフォルダは **ツール専用の独立リポジトリ**として使う想定です。素材（画像など）は別フォルダに置いたまま、**パス参照**で運用してください。

## セットアップ

### 1) 依存を入れる

```bash
pip install -r requirements.txt
```

### 2) APIキーを設定（おすすめ：.env）

`env.example` をコピーして `.env` を作ってください。

```bash
cp env.example .env
```

`.env` は `.gitignore` されるので、**APIキーがコミットされません**。

## 使い方

### まずは1枚生成

```bash
python run_gemini3_images.py \
  --prompt "白背景、商品の物撮り写真、自然光、超高精細" \
  --candidate_count 1
```

### 画像を入力に混ぜる（フォルダ/単体/混在OK）

`gpt52-image-cli` と同じ感覚で、入力画像を混ぜられます（再帰探索・重複除去あり）。

```bash
python run_gemini3_images.py \
  --images_dir "/path/to/styleguide_pages" \
  --max_images 20 \
  --image "/path/to/creative.jpeg" \
  --prompt "入力画像を参考に、同じ雰囲気の新しい画像を生成して。" \
  --candidate_count 1
```

### スタイル画像で「絵柄/タッチ」を指定する（--style）

`--style` に画像（ファイル）やフォルダを渡すと、**絵柄/スタイルの参照画像**としてリクエストに添付されます。

- **フォルダ/単体/混在OK**（`--style` は複数回指定できます）
- **添付順は「通常の入力画像 → スタイル画像」**で固定されます
- スタイル画像がある場合、プロンプトの末尾に自動で以下が追加されます（番号は添付順と一致）：

```text
# **追加指示: これらの画像の絵柄/スタイルに忠実に従った画像を生成してください**
- imageX
- imageY
```

例：

```bash
python run_gemini3_images.py \
  --prompt "サウナはなぜ気持ちいいのか図解してください" \
  --style "/path/to/style_images" \
  --candidate_count 1
```

フォルダ＋単体の混在例：

```bash
python run_gemini3_images.py \
  --prompt "添付画像の雰囲気を維持して新しい図を作って" \
  --images_dir "/path/to/reference_images" \
  --style "/path/to/style_images" "/path/to/extra_style.jpg" \
  --candidate_count 1
```

※ スタイル画像は最大10枚です。11枚以上ある場合は **警告を出して先頭10枚のみ**使用します。

### 複数枚生成（candidateCount）

```bash
python run_gemini3_images.py \
  --prompt "白背景、化粧品の物撮り、柔らかい影、プロ品質" \
  --candidate_count 3 \
  --seed 42 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 40
```

※ 一部のモデルは「1リクエストで複数候補（candidateCount>1）」が無効です。その場合このCLIは **内部的に 1枚×N回** で実現します（出力は同じフォルダにまとまります）。

### 画像設定（imageConfig）

```bash
python run_gemini3_images.py \
  --prompt "正方形、EC用、白背景、商品写真" \
  --candidate_count 2 \
  --aspect_ratio "1:1" \
  --image_size "1024x1024"
```

## 出力

- 生成結果は `output/YYYYMMDD_HHMMSS_ミリ秒/` に保存されます
- 画像ファイル: `image_01_01.png` のような名前で保存されます
- デバッグ用に `request.json` / `response.json` / `meta.json` も同じフォルダに保存されます（`--no_save_json` で無効化できます）

## モデル名について（重要）

Gemini側のモデル名は更新されることがあります。もしエラーが出る場合は、まず `--model` を公式ドキュメントのモデル名に合わせて指定してください。

モデル名が分からない場合は、次で一覧を確認できます：

```bash
python run_gemini3_images.py --list_models
```


