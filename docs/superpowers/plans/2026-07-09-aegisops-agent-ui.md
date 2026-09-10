# AegisOps Agent UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the existing static chat home screen to match the supplied AegisOps Agent reference while preserving current chat behavior.

**Architecture:** Keep the FastAPI static frontend shape intact. Add static regression tests, replace the landing markup and CSS in place, and make JavaScript/config naming changes without changing API contracts.

**Tech Stack:** FastAPI static files, vanilla HTML/CSS/JavaScript, pytest.

---

### Task 1: Static Branding Regression Test

**Files:**
- Create: `tests/unit/test_static_aegisops_ui.py`
- Read: `static/index.html`
- Read: `static/app.js`
- Read: `app/config.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_static_home_uses_aegisops_branding_and_reference_sections():
    index_html = read_text("static/index.html")

    assert "<title>AegisOps Agent</title>" in index_html
    assert "AegisOps Agent" in index_html
    assert "你好！我是 <span>AegisOps Agent</span>" in index_html
    assert "AegisOps 助力运维" in index_html
    assert "分析系统性能瓶颈" in index_html
    assert "查询错误日志" in index_html
    assert "生成巡检报告" in index_html
    assert "推荐优化方案" in index_html
    assert "问问 AegisOps Agent..." in index_html


def test_runtime_name_defaults_to_aegisops_agent():
    config_py = read_text("app/config.py")
    app_js = read_text("static/app.js")

    assert 'app_name: str = "AegisOps Agent"' in config_py
    assert "class AegisOpsAgentApp" in app_js
    assert "new AegisOpsAgentApp()" in app_js
    old_product_name = "Super" + "BizAgent"
    assert old_product_name not in app_js
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_static_aegisops_ui.py -q`

Expected: `FAIL` because the existing static files still use the previous product name and do not contain the reference home-state sections.

### Task 2: HTML And CSS Redesign

**Files:**
- Modify: `static/index.html`
- Modify: `static/styles.css`

- [ ] **Step 1: Replace home shell markup**

Use the existing element IDs (`newChatBtn`, `aiOpsSidebarBtn`, `welcomeGreeting`, `chatMessages`, `messageInput`, `toolsBtn`, `uploadFileItem`, `modeSelectorBtn`, `modeDropdown`, `sendButton`, `fileInput`) so the current JavaScript continues to bind.

- [ ] **Step 2: Replace static styling**

Implement the warm ivory/orange visual system, sidebar, history cards, central CSS/SVG assistant illustration, quick action chips, input panel, chat message states, dropdowns, and responsive behavior.

- [ ] **Step 3: Run the static test**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_static_aegisops_ui.py -q`

Expected: only the runtime name assertions still fail until Task 3 is complete.

### Task 3: JavaScript And Backend Name Sync

**Files:**
- Modify: `static/app.js`
- Modify: `app/config.py`

- [ ] **Step 1: Rename frontend class and visible runtime strings**

Rename the previous frontend app class to `AegisOpsAgentApp`, update the placeholder string to `问问 AegisOps Agent...`, and keep all method names and element IDs stable.

- [ ] **Step 2: Update backend default app name**

Change the previous default `app_name` value to `AegisOps Agent`.

- [ ] **Step 3: Run the static test**

Run: `.venv\Scripts\python.exe -m pytest tests/unit/test_static_aegisops_ui.py -q`

Expected: `PASS`.

### Task 4: Runtime Verification

**Files:**
- Read: `static/index.html`
- Read: `static/styles.css`
- Read: `static/app.js`

- [ ] **Step 1: Run a syntax check for JavaScript**

Run: `node --check static/app.js`

Expected: no syntax errors. If Node is unavailable, record that limitation.

- [ ] **Step 2: Start local server**

Run: `.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 9900`

Expected: the server starts and serves `/`.

- [ ] **Step 3: Open browser and capture screenshot**

Open `http://127.0.0.1:9900/`, capture a screenshot, and verify the empty state resembles the supplied reference: orange left sidebar, AegisOps Agent wordmark, AI Ops pill, centered assistant illustration, greeting, quick chips, and large rounded input.

### Commit Note

This workspace is not a Git repository, so commit steps are intentionally omitted.
