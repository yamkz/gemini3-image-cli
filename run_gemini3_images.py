import argparse
import base64
import mimetypes
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests


# ListModels (v1beta) で見えるモデル名をデフォルトにする
DEFAULT_MODEL = "gemini-3-pro-image-preview"

IMAGE_EXTS_DEFAULT = (".png", ".jpg", ".jpeg", ".webp", ".gif")


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def load_dotenv_if_present(dotenv_path: Path | None = None) -> None:
    """
    Minimal .env loader (no external dependency).
    - Supports KEY=VALUE lines
    - Ignores blank lines and comments starting with '#'
    - Does not overwrite already-set environment variables
    """
    p = dotenv_path or (Path.cwd() / ".env")
    if not p.exists() or not p.is_file():
        return
    try:
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        return


def _timestamp_dirname() -> str:
    # YYYYMMDD_HHMMSS_mmm (milliseconds)
    now = datetime.now()
    ms = int(now.microsecond / 1000)
    return now.strftime("%Y%m%d_%H%M%S_") + f"{ms:03d}"


def _ext_from_mime(mime: str | None) -> str:
    if not mime:
        return "png"
    m = mime.lower()
    if "png" in m:
        return "png"
    if "jpeg" in m or "jpg" in m:
        return "jpg"
    if "webp" in m:
        return "webp"
    if "gif" in m:
        return "gif"
    return "bin"


def _mime_for_path(p: Path) -> str:
    mime, _ = mimetypes.guess_type(str(p))
    return mime or "image/jpeg"


def collect_images(images_dir: Path, max_images: int | None = None) -> list[Path]:
    paths: list[Path] = []
    for p in sorted(images_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS_DEFAULT:
            paths.append(p)
            if max_images is not None and len(paths) >= max_images:
                break
    return paths


def normalize_and_dedupe(paths: list[Path]) -> list[Path]:
    # Deduplicate by resolved absolute path, keep first occurrence order,
    # then return stable sorted by path string for repeatable behavior.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in paths:
        rp = str(p.expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            uniq.append(Path(rp))
    return sorted(uniq, key=lambda x: str(x))


def build_parts(prompt: str, image_paths: list[Path]) -> list[dict]:
    parts: list[dict] = [{"text": prompt}]
    for p in image_paths:
        data_b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        parts.append(
            {
                "inlineData": {
                    "mimeType": _mime_for_path(p),
                    "data": data_b64,
                }
            }
        )
    return parts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate image(s) with Gemini (REST) and save inlineData.data (base64) to local files."
    )
    p.add_argument("--list_models", action="store_true", help="利用可能なモデル一覧を表示して終了する")
    p.add_argument("--prompt", type=str, required=False, help="画像生成プロンプト（自由）")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="モデル名（公式ドキュメントに合わせて変更可）")
    p.add_argument("--images_dir", type=str, help="画像フォルダのパス（配下を再帰検索）")
    p.add_argument(
        "--image",
        action="append",
        default=[],
        help="画像ファイルを個別指定（複数OK、複数回指定可）",
    )
    p.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="画像ファイルを複数パスで指定（例: --images a.jpg b.png）",
    )
    p.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="フォルダから拾う最大枚数（大量画像の安全策）",
    )
    p.add_argument("--candidate_count", type=int, default=1, help="生成する画像の枚数（candidateCount）")
    p.add_argument("--seed", type=int, default=None, help="seed（未指定なら送らない）")
    p.add_argument("--temperature", type=float, default=None, help="temperature（未指定なら送らない）")
    p.add_argument("--top_p", type=float, default=None, help="topP（未指定なら送らない）")
    p.add_argument("--top_k", type=int, default=None, help="topK（未指定なら送らない）")
    p.add_argument("--aspect_ratio", type=str, default=None, help="imageConfig.aspectRatio（例: 1:1, 4:3, 16:9）")
    p.add_argument("--image_size", type=str, default=None, help="imageConfig.imageSize（例: 1024x1024 など）")
    p.add_argument("--out_root", type=str, default="output", help="出力ルート（デフォルト: output）")
    p.add_argument("--timeout", type=int, default=300, help="HTTPタイムアウト（秒）")
    p.add_argument("--no_save_json", action="store_true", help="request/response JSONを保存しない")
    return p.parse_args()


def build_generation_config(args: argparse.Namespace) -> dict:
    gc: dict = {"candidateCount": args.candidate_count}

    if args.seed is not None:
        gc["seed"] = args.seed
    if args.temperature is not None:
        gc["temperature"] = args.temperature
    if args.top_p is not None:
        gc["topP"] = args.top_p
    if args.top_k is not None:
        gc["topK"] = args.top_k

    img_cfg: dict = {}
    if args.aspect_ratio:
        img_cfg["aspectRatio"] = args.aspect_ratio
    if args.image_size:
        img_cfg["imageSize"] = args.image_size
    if img_cfg:
        gc["imageConfig"] = img_cfg

    return gc


def post_generate_content(api_key: str, model: str, payload: dict, timeout_s: int) -> requests.Response:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    return requests.post(url, headers=headers, json=payload, timeout=timeout_s)


def list_models(api_key: str, timeout_s: int) -> list[dict]:
    """
    ListModels API (v1beta) を叩いて、モデル一覧を返す。
    返り値はレスポンスの "models" 配列をページネーション込みで集めたもの。
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    all_models: list[dict] = []
    page_token: str | None = None
    for _ in range(50):  # safety
        params = {}
        if page_token:
            params["pageToken"] = page_token
        resp = requests.get(url, headers=headers, params=params, timeout=timeout_s)
        if resp.status_code != 200:
            raise RuntimeError(f"ListModels failed: {resp.status_code} {resp.text}")
        data = resp.json()
        all_models.extend(data.get("models", []) or [])
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return all_models


def try_generate(api_key: str, args: argparse.Namespace, base_payload: dict) -> tuple[dict, dict] | tuple[None, dict]:
    """
    Gemini側の互換性対策:
    - responseModalities / responseModality は環境によって未対応なことがあるため、
      「付ける/付けない」を含めて順に試す。
    Returns: (response_json, used_payload) or (None, error_info)
    """
    gen_cfg = base_payload.get("generationConfig", {}) or {}

    variants: list[dict] = []

    # v1: responseModalities: ["IMAGE"]
    v1 = dict(base_payload)
    v1["generationConfig"] = dict(gen_cfg)
    v1["generationConfig"]["responseModalities"] = ["IMAGE"]
    variants.append(v1)

    # v2: responseModality: "IMAGE"
    v2 = dict(base_payload)
    v2["generationConfig"] = dict(gen_cfg)
    v2["generationConfig"]["responseModality"] = "IMAGE"
    variants.append(v2)

    # v0: no response modality fields (model/output defaultsに任せる)
    v0 = dict(base_payload)
    v0["generationConfig"] = dict(gen_cfg)
    v0["generationConfig"].pop("responseModalities", None)
    v0["generationConfig"].pop("responseModality", None)
    variants.append(v0)

    errors: list[dict] = []
    for payload in variants:
        try:
            resp = post_generate_content(api_key=api_key, model=args.model, payload=payload, timeout_s=args.timeout)
        except Exception as e:
            errors.append({"type": "exception", "message": str(e), "payload": payload})
            continue

        if resp.status_code == 200:
            return resp.json(), payload

        errors.append(
            {
            "type": "http_error",
            "status": resp.status_code,
            "text": resp.text,
            "payload": payload,
            }
        )

    return None, {"type": "all_attempts_failed", "attempts": errors}


def extract_inline_images(resp_json: dict) -> list[dict]:
    """
    candidates[*].content.parts[*].inlineData.data を集める。
    返り値: [{"b64": "...", "mime": "image/png", "i": 1, "j": 1}, ...]
    """
    out: list[dict] = []
    candidates = resp_json.get("candidates", []) or []
    for i, cand in enumerate(candidates, start=1):
        content = cand.get("content", {}) or {}
        parts = content.get("parts", []) or []
        for j, part in enumerate(parts, start=1):
            inline = part.get("inlineData") or part.get("inline_data")
            if not isinstance(inline, dict):
                continue
            b64 = inline.get("data")
            if not b64:
                continue
            mime = inline.get("mimeType") or inline.get("mime_type")
            out.append({"b64": b64, "mime": mime, "i": i, "j": j})
    return out


def main() -> int:
    args = parse_args()

    # Allow using a local .env file (kept out of git) for GEMINI_API_KEY.
    load_dotenv_if_present()

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        _eprint("APIキーが見つかりません。.env に GEMINI_API_KEY=... を入れてください（または環境変数）。")
        return 2

    if args.list_models:
        try:
            models = list_models(api_key=api_key, timeout_s=args.timeout)
        except Exception as e:
            _eprint(f"モデル一覧の取得に失敗しました: {e}")
            return 1

        # 見やすく：generateContent対応＋名前にimageを含むものを優先表示
        def supports_generate(m: dict) -> bool:
            methods = m.get("supportedGenerationMethods", []) or []
            return "generateContent" in methods

        image_like = []
        others = []
        for m in models:
            name = (m.get("name") or "").lower()
            if "image" in name:
                image_like.append(m)
            else:
                others.append(m)

        def print_group(title: str, group: list[dict]) -> None:
            print(f"\n== {title} ==")
            for m in group:
                if not supports_generate(m):
                    continue
                print(m.get("name"))

        print_group("NAME contains 'image' (generateContent supported)", image_like)
        print_group("Others (generateContent supported)", others)
        print("\nTip: 生成に使う時は --model の値として 'models/...' ではなく、'gemini-...' の部分だけを渡します（例: nameが 'models/gemini-1.5-pro' なら --model gemini-1.5-pro）。")
        return 0

    if not args.prompt:
        _eprint("--prompt を指定してください。（または --list_models を使ってモデル一覧を確認できます）")
        return 2

    image_paths: list[Path] = []
    if args.images_dir:
        d = Path(args.images_dir).expanduser()
        if not d.exists() or not d.is_dir():
            _eprint(f"images_dir が存在しないかフォルダではありません: {d}")
            return 2
        image_paths.extend(collect_images(d, max_images=args.max_images))

    for ip in args.image:
        p = Path(ip).expanduser()
        if not p.exists() or not p.is_file():
            _eprint(f"image が存在しないかファイルではありません: {p}")
            return 2
        image_paths.append(p)

    for ip in args.images:
        p = Path(ip).expanduser()
        if not p.exists() or not p.is_file():
            _eprint(f"images が存在しないかファイルではありません: {p}")
            return 2
        image_paths.append(p)

    image_paths = normalize_and_dedupe(image_paths)

    started = time.time()

    out_dir = Path(args.out_root) / _timestamp_dirname()
    out_dir.mkdir(parents=True, exist_ok=True)

    def run_once(request_index: int, seed_override: int | None) -> tuple[int, int] | tuple[None, dict]:
        """
        1回分の生成を実行して保存する。
        Returns: (saved_count, elapsed_ms) or (None, error_info)
        """
        gc = build_generation_config(args)
        # 1回のリクエストでは candidateCount は 1 に固定（複数候補が無効なモデルがあるため）
        gc["candidateCount"] = 1
        if seed_override is not None:
            gc["seed"] = seed_override
        base_payload = {
            "contents": [{"parts": build_parts(args.prompt, image_paths)}],
            "generationConfig": gc,
        }

        _eprint(f"Requesting image generation... ({request_index}/{args.candidate_count}) model={args.model}")
        t0 = time.time()
        resp_json, used_payload_or_err = try_generate(api_key=api_key, args=args, base_payload=base_payload)
        if resp_json is None:
            return None, used_payload_or_err

        used_payload = used_payload_or_err
        if not args.no_save_json:
            (out_dir / f"request_{request_index:02d}.json").write_text(
                json.dumps(used_payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (out_dir / f"response_{request_index:02d}.json").write_text(
                json.dumps(resp_json, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        images = extract_inline_images(resp_json)
        saved_local = 0
        for item in images:
            b64 = item["b64"]
            mime = item.get("mime")
            ext = _ext_from_mime(mime)
            try:
                raw = base64.b64decode(b64)
            except Exception:
                continue
            path = out_dir / f"image_{request_index:02d}_{item['i']:02d}_{item['j']:02d}.{ext}"
            path.write_bytes(raw)
            print(str(path))
            saved_local += 1

        return saved_local, int((time.time() - t0) * 1000)

    # candidate_count は「欲しい画像枚数」として扱い、必要なら複数回叩いて実現する。
    saved = 0
    per_request_ms: list[int] = []
    for idx in range(1, args.candidate_count + 1):
        seed_override = None
        if args.seed is not None:
            seed_override = args.seed + (idx - 1)
        result = run_once(request_index=idx, seed_override=seed_override)
        if result[0] is None:
            err = result[1]
            _eprint("API呼び出しに失敗しました。")
            if not args.no_save_json:
                (out_dir / f"error_{idx:02d}.json").write_text(
                    json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            _eprint("ヒント: まず --model を公式ドキュメントのモデル名に合わせてください。")
            _eprint(f"エラー詳細は {out_dir} を確認してください。")
            return 1
        saved_local, ms = result
        saved += saved_local
        per_request_ms.append(ms)

    elapsed_ms = int((time.time() - started) * 1000)
    meta = {
        "model": args.model,
        "candidateCountRequested": args.candidate_count,
        "requestsMade": args.candidate_count,
        "savedImages": saved,
        "outputDir": str(out_dir),
        "elapsedMs": elapsed_ms,
        "perRequestMs": per_request_ms,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if saved == 0:
        _eprint("画像が保存できませんでした（inlineData が無い/形式が違う可能性）。")
        _eprint(f"response.json を確認してください: {out_dir}")
        return 1

    _eprint(f"Saved {saved} image(s) to: {out_dir}  ({elapsed_ms} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


