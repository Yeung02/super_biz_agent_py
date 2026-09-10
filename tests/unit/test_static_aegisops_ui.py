from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_static_home_uses_aegisops_branding_and_reference_sections():
    index_html = read_text("static/index.html")

    assert "<title>AegisOps Agent</title>" in index_html
    assert 'rel="icon"' in index_html
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
    assert "renderEmptyHistoryExamples" in app_js
    assert "CPU占用率很高怎么办？" in app_js
    old_product_name = "Super" + "BizAgent"
    assert old_product_name not in app_js


def test_frontend_exposes_backend_capabilities_as_actions():
    index_html = read_text("static/index.html")
    app_js = read_text("static/app.js")

    for element_id in (
        'id="viewAllHistoryBtn"',
        'id="opsLearnMoreBtn"',
        'id="settingsBtn"',
        'id="helpBtn"',
        'id="operationsPanel"',
        'id="healthCheckBtn"',
        'id="indexDirectoryBtn"',
        'id="indexDirectoryInput"',
        'id="uploadFromPanelBtn"',
        'id="triggerAIOpsPanelBtn"',
        'id="clearSessionBtn"',
    ):
        assert element_id in index_html

    assert "runHealthCheck()" in app_js
    assert "runDirectoryIndex()" in app_js
    assert "openOperationsPanel(" in app_js
    assert "renderHistoryPanel()" in app_js
    assert "clearCurrentSession()" in app_js
    assert "this.apiBaseUrl = '/api';" in app_js
    assert "fetch(`${this.apiBaseUrl}/health`" in app_js
    assert "fetch(`${this.apiBaseUrl}/file/index_directory`" in app_js
    assert "fetch('/api/chat/clear'" in app_js


def test_static_assets_are_cache_busted_for_port_reuse():
    index_html = read_text("static/index.html")

    assert 'href="/static/styles.css?v=aegisops-agent-history-db"' in index_html
    assert 'src="/static/app.js?v=aegisops-agent-history-db"' in index_html


def test_upload_toolbar_button_opens_file_picker_directly():
    index_html = read_text("static/index.html")
    app_js = read_text("static/app.js")

    assert 'id="toolsBtn"' in index_html
    assert 'id="fileInput"' in index_html
    assert "openFilePicker()" in app_js
    assert "this.toolsBtn.addEventListener('click'" in app_js
    assert "this.openFilePicker();" in app_js


def test_upload_request_normalizes_supported_document_mime_types():
    app_js = read_text("static/app.js")

    assert "normalizeUploadFile(file)" in app_js
    assert "'.md': 'text/markdown'" in app_js
    assert "'.markdown': 'text/markdown'" in app_js
    assert "'.txt': 'text/plain'" in app_js
    assert "formData.append('file', uploadFile);" in app_js


def test_all_conversations_panel_loads_backend_session_list():
    app_js = read_text("static/app.js")

    assert "loadBackendChatHistories()" in app_js
    assert "fetch(`${this.apiBaseUrl}/chat/sessions`" in app_js
    assert "backendHistorySource" in app_js
