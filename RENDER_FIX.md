# 🔧 Render Deployment Fix — Root Cause + Step-by-Step Solution

---

## Why Render Ignores Your Python Version (Root Cause Analysis)

### Problem 1: `runtime.txt` format is wrong
Render requires a **very specific format**. Any variation = ignored silently = falls back to default (Python 3.14).

```
# ❌ WRONG formats (all silently ignored by Render)
3.10
3.10.13
Python 3.10
python 3.10
python3.10

# ✅ ONLY format that works
python-3.10.13
```

### Problem 2: `runtime.txt` is NOT in the repo root
Render only reads `runtime.txt` from the **root of your repository**.

```
# ❌ WRONG — Render won't find this
/my-project/backend/runtime.txt

# ✅ CORRECT — must be at root level
/runtime.txt   (same level as render.yaml)
```

### Problem 3: `render.yaml` pythonVersion does NOT control Python
The `pythonVersion` field in `render.yaml` is **display metadata only** on most Render plans.
The only reliable control is `runtime.txt`.

### Problem 4: pandas/numpy building from source
When Python 3.14 is used, pip can't find prebuilt wheels (PyPI doesn't have cp314 wheels yet),
so pip falls back to downloading `.tar.gz` source and compiling — which fails because
`setuptools.build_meta` isn't available in the build environment.

Fix: Force Python 3.10, which has full wheel support for all scientific packages.

---

## Step-by-Step Fix

### Step 1 — Verify your repo structure
Your repository root **must** look like this:

```
your-repo/
├── runtime.txt          ← MUST be here (root level)
├── render.yaml          ← MUST be here (root level)
├── requirements.txt     ← MUST be here (root level)
├── main.py
├── ...other .py files
```

If your Python files are in a subdirectory (e.g., `backend/`), you have two options:
- Option A: Move `runtime.txt` to repo root anyway (Render reads it globally)
- Option B: Set root directory in Render dashboard → Settings → "Root Directory" = `backend`
  Then put `runtime.txt` inside `backend/`

### Step 2 — Create `runtime.txt` with exact content

```bash
# Run this in your repo root
echo "python-3.10.13" > runtime.txt

# Verify it (must show exactly this, one line, no extra spaces)
cat runtime.txt
# python-3.10.13
```

### Step 3 — Update `render.yaml`

```yaml
services:
  - type: web
    name: ai-trading-engine
    env: python
    pythonVersion: "3.10.13"
    plan: free
    buildCommand: pip install --upgrade pip setuptools wheel && pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    autoDeploy: true
    envVars:
      - key: ENV
        value: production
```

**Key change in `buildCommand`**: Always run `pip install --upgrade pip setuptools wheel` FIRST.
This fixes the `Cannot import 'setuptools.build_meta'` error.

### Step 4 — Pin exact package versions in `requirements.txt`

Use only versions with confirmed `cp310-manylinux_2_17_x86_64` wheels:

```
pandas==2.1.4       ← NOT 2.2.x (edge cases on py310)
numpy==1.26.4       ← confirmed cp310 wheel
scikit-learn==1.4.2 ← confirmed cp310 wheel
xgboost==2.0.3      ← confirmed cp310 wheel
lightgbm==4.3.0     ← confirmed cp310 wheel
```

### Step 5 — Commit and trigger a clean build

```bash
git add runtime.txt render.yaml requirements.txt
git commit -m "fix: force Python 3.10, pin wheel-only deps"
git push origin main
```

Then in Render dashboard:
1. Go to your service → **Settings** → **Build & Deploy**
2. Click **"Clear build cache & deploy"** (not just "Manual Deploy")
3. This forces Render to re-read `runtime.txt` from scratch

### Step 6 — Verify Python version in build logs

In Render build logs, look for this line near the top:
```
==> Using Python version: 3.10.13 (from runtime.txt)
```

If you still see `3.14.x`, the `runtime.txt` file is either:
- In the wrong location
- Has wrong format (re-check Step 2)
- You need to set "Root Directory" in Render dashboard

---

## If `runtime.txt` Still Doesn't Work

Some Render accounts are on newer infrastructure that reads `render.yaml` only.
Try this alternative — set Python version directly in `render.yaml` build command:

```yaml
buildCommand: >
  curl https://pyenv.run | bash &&
  export PATH="$HOME/.pyenv/bin:$PATH" &&
  pyenv install 3.10.13 &&
  pyenv local 3.10.13 &&
  pip install --upgrade pip setuptools wheel &&
  pip install -r requirements.txt
```

OR (simpler) — use Render's environment variable approach:

In Render Dashboard → Environment → Add:
```
PYTHON_VERSION = 3.10.13
```

---

## Quick Diagnosis Checklist

Run these checks before deploying:

```bash
# 1. Is runtime.txt at repo root?
ls -la runtime.txt

# 2. Is the content exactly right?
cat runtime.txt
# Must print: python-3.10.13

# 3. No trailing spaces or Windows line endings?
cat -A runtime.txt
# Must print: python-3.10.13$   (just $ at end, no ^M)

# If you see ^M (Windows CRLF), fix it:
sed -i 's/\r//' runtime.txt

# 4. Is requirements.txt valid?
pip install --dry-run -r requirements.txt 2>&1 | grep -i "error"
# Should be empty (no errors)
```

---

## Render Free Tier Memory Limits

| Package | RAM usage | Safe? |
|---------|-----------|-------|
| fastapi + uvicorn | ~50MB | ✅ |
| pandas + numpy | ~80MB | ✅ |
| scikit-learn | ~30MB | ✅ |
| xgboost | ~40MB | ✅ |
| lightgbm | ~40MB | ✅ |
| matplotlib | ~40MB | ✅ |
| **Total** | **~280MB** | ✅ Free tier (512MB) |
| torch (if added) | ~1.8GB | ❌ CRASH |
| transformers (if added) | ~600MB | ❌ CRASH |

---

## Summary of Files

| File | Location | Purpose |
|------|----------|---------|
| `runtime.txt` | repo root | Forces Python 3.10.13 |
| `.python-version` | repo root | Fallback for newer Render stacks |
| `render.yaml` | repo root | Deployment config with upgraded build command |
| `requirements.txt` | repo root | Pinned wheel-compatible versions |
