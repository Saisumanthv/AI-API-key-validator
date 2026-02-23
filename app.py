import streamlit as st
import requests

st.set_page_config(page_title="AI API Key Validator", page_icon="🔑", layout="centered")

st.title("🔑 AI API Key Validator")
st.caption("Enter your key → fetch available models → pick one → see the real AI response.")

TEST_PROMPT = "Is this API key working? Please give a short, fun confirmation that it is."

# ── Model filter helpers ───────────────────────────────────────────────────────

def is_chat_model_openai(mid):
    mid = mid.lower()
    return any(x in mid for x in ["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt"]) and "instruct" not in mid

def is_chat_model_groq(name):
    return any(x in name.lower() for x in ["llama", "mixtral", "gemma", "qwen", "deepseek", "mistral"])

def is_chat_model_together(m):
    n = (m.get("display_name", "") + m.get("id", "")).lower()
    t = m.get("type", "").lower()
    return "chat" in t or "instruct" in n or "chat" in n

def is_chat_model_mistral(name):
    return not any(s in name.lower() for s in ["embed", "moderation"])

def is_chat_model_cohere(name):
    return any(x in name.lower() for x in ["command", "c4ai"])

# ── Step 1: Fetch model list only ─────────────────────────────────────────────
# Each returns (success: bool, models: list[str], error: str|None)

def fetch_models_openai(key):
    try:
        import openai
        client = openai.OpenAI(api_key=key)
        all_models = list(client.models.list())
        chat_models = sorted([m.id for m in all_models if is_chat_model_openai(m.id)])
        return True, chat_models or ["gpt-3.5-turbo"], None
    except openai.AuthenticationError:
        return False, [], "Invalid API key. Authentication failed."
    except Exception as e:
        return False, [], str(e)

def fetch_models_anthropic(key):
    ANTHROPIC_MODELS = [
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229", "claude-3-haiku-20240307",
    ]
    try:
        import anthropic
        anthropic.Anthropic(api_key=key)  # validates key without a call
        # Do a cheap ping to verify key works
        client = anthropic.Anthropic(api_key=key)
        client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True, ANTHROPIC_MODELS, None
    except anthropic.AuthenticationError:
        return False, [], "Invalid API key. Authentication failed."
    except Exception as e:
        return False, [], str(e)

def fetch_models_gemini(key):
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        all_models = list(genai.list_models())
        chat_models = [
            m.name for m in all_models
            if "generateContent" in (m.supported_generation_methods or [])
        ]
        return True, chat_models or ["models/gemini-1.5-flash"], None
    except Exception as e:
        err = str(e)
        if "api key" in err.lower() or "invalid" in err.lower() or "401" in err:
            return False, [], "Invalid API key. Authentication failed."
        return False, [], err

def fetch_models_groq(key):
    try:
        from groq import Groq
        client = Groq(api_key=key)
        all_models = client.models.list().data
        chat_models = [m.id for m in all_models if is_chat_model_groq(m.id)]
        return True, chat_models or [m.id for m in all_models], None
    except Exception as e:
        err = str(e)
        if "invalid" in err.lower() or "auth" in err.lower() or "401" in err:
            return False, [], "Invalid API key. Authentication failed."
        return False, [], err

def fetch_models_mistral(key):
    try:
        from mistralai import Mistral
        client = Mistral(api_key=key)
        all_models = client.models.list().data
        chat_models = [m.id for m in all_models if is_chat_model_mistral(m.id)]
        return True, chat_models or [m.id for m in all_models], None
    except Exception as e:
        err = str(e)
        if "unauthorized" in err.lower() or "invalid" in err.lower() or "401" in err:
            return False, [], "Invalid API key. Authentication failed."
        return False, [], err

def fetch_models_cohere(key):
    try:
        headers = {"Authorization": f"Bearer {key}"}
        r = requests.get("https://api.cohere.com/v1/models", headers=headers)
        if r.status_code == 401:
            return False, [], "Invalid API key. Authentication failed."
        if r.status_code == 200:
            all_models = [m["name"] for m in r.json().get("models", [])]
            chat_models = [m for m in all_models if is_chat_model_cohere(m)]
            return True, chat_models or all_models[:5] or ["command-r"], None
        return True, ["command-r"], None
    except Exception as e:
        return False, [], str(e)

def fetch_models_together(key):
    headers = {"Authorization": f"Bearer {key}"}
    r = requests.get("https://api.together.xyz/v1/models", headers=headers)
    if r.status_code == 401:
        return False, [], "Invalid API key. Authentication failed."
    if r.status_code != 200:
        return False, [], f"Error {r.status_code}"
    all_models = r.json()
    chat_models = [m["id"] for m in all_models if is_chat_model_together(m)]
    return True, chat_models or [m["id"] for m in all_models[:10]], None

def fetch_models_perplexity(key):
    PERPLEXITY_MODELS = [
        "sonar-pro", "sonar", "sonar-reasoning-pro", "sonar-reasoning",
        "r1-1776", "llama-3.1-sonar-large-128k-online",
    ]
    # Verify key with a tiny call
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json={"model": "sonar", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        headers=headers,
    )
    if resp.status_code == 401:
        return False, [], "Invalid API key. Authentication failed."
    return True, PERPLEXITY_MODELS, None

def fetch_models_huggingface(key):
    headers = {"Authorization": f"Bearer {key}"}
    whoami = requests.get("https://huggingface.co/api/whoami", headers=headers)
    if whoami.status_code == 401:
        return False, [], "Invalid API key. Authentication failed."
    r = requests.get(
        "https://huggingface.co/api/models",
        params={"pipeline_tag": "text-generation", "sort": "downloads", "limit": 30},
        headers=headers,
    )
    model_names = [m["id"] for m in r.json()] if r.status_code == 200 else ["gpt2"]
    return True, model_names, None

def fetch_models_sarvam(key):
    SARVAM_MODELS = ["mayura:v1"]
    headers = {"api-subscription-key": key, "Content-Type": "application/json"}
    resp = requests.post(
        "https://api.sarvam.ai/translate",
        json={"input": "hi", "source_language_code": "en-IN", "target_language_code": "hi-IN",
              "model": "mayura:v1", "enable_preprocessing": False},
        headers=headers,
    )
    if resp.status_code in (401, 403):
        return False, [], "Invalid API key. Authentication failed."
    return True, SARVAM_MODELS, None


# ── Step 2: Run the actual chat call with the chosen model ────────────────────
# Each returns (success: bool, response_text: str, error: str|None)

def call_openai(key, model):
    try:
        import openai
        client = openai.OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": TEST_PROMPT}], max_tokens=200,
        )
        return True, resp.choices[0].message.content.strip(), None
    except Exception as e:
        return False, None, str(e)

def call_anthropic(key, model):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model, max_tokens=200, messages=[{"role": "user", "content": TEST_PROMPT}],
        )
        return True, resp.content[0].text.strip(), None
    except Exception as e:
        return False, None, str(e)

def call_gemini(key, model):
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        m = genai.GenerativeModel(model)
        resp = m.generate_content(TEST_PROMPT)
        return True, resp.text.strip(), None
    except Exception as e:
        return False, None, str(e)

def call_groq(key, model):
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": TEST_PROMPT}], max_tokens=200,
        )
        return True, resp.choices[0].message.content.strip(), None
    except Exception as e:
        return False, None, str(e)

def call_mistral(key, model):
    try:
        from mistralai import Mistral
        client = Mistral(api_key=key)
        resp = client.chat.complete(
            model=model, messages=[{"role": "user", "content": TEST_PROMPT}], max_tokens=200,
        )
        return True, resp.choices[0].message.content.strip(), None
    except Exception as e:
        return False, None, str(e)

def call_cohere(key, model):
    try:
        import cohere
        client = cohere.ClientV2(api_key=key)
        resp = client.chat(
            model=model, messages=[{"role": "user", "content": TEST_PROMPT}], max_tokens=200,
        )
        return True, resp.message.content[0].text.strip(), None
    except Exception as e:
        return False, None, str(e)

def call_together(key, model):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = requests.post(
        "https://api.together.xyz/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": TEST_PROMPT}], "max_tokens": 200},
        headers=headers,
    )
    if resp.status_code == 200:
        return True, resp.json()["choices"][0]["message"]["content"].strip(), None
    return False, None, f"Error {resp.status_code}: {resp.text[:200]}"

def call_perplexity(key, model):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    resp = requests.post(
        "https://api.perplexity.ai/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": TEST_PROMPT}], "max_tokens": 200},
        headers=headers,
    )
    if resp.status_code == 200:
        return True, resp.json()["choices"][0]["message"]["content"].strip(), None
    return False, None, f"Error {resp.status_code}: {resp.text[:200]}"

def call_huggingface(key, model):
    headers = {"Authorization": f"Bearer {key}"}
    payload = {"inputs": TEST_PROMPT, "parameters": {"max_new_tokens": 100, "return_full_text": False}}
    resp = requests.post(
        f"https://api-inference.huggingface.co/models/{model}",
        json=payload, headers=headers, timeout=20,
    )
    if resp.status_code == 200:
        data = resp.json()
        if isinstance(data, list) and data:
            text = data[0].get("generated_text", "").strip()
            return True, text or "(Model returned empty output — try another model)", None
    elif resp.status_code == 503:
        return False, None, "Model is loading on HuggingFace servers. Wait ~20s and retry."
    return False, None, f"Error {resp.status_code}: {resp.text[:200]}"

def call_sarvam(key, model):
    headers = {"api-subscription-key": key, "Content-Type": "application/json"}
    resp = requests.post(
        "https://api.sarvam.ai/translate",
        json={"input": "Is this API key working? Give a short confirmation.",
              "source_language_code": "en-IN", "target_language_code": "hi-IN",
              "model": model, "enable_preprocessing": False},
        headers=headers,
    )
    if resp.status_code == 200:
        translated = resp.json().get("translated_text", "")
        return True, f"Yes! Sarvam AI key is working 🎉\n\nTest translation → **{translated}**", None
    return False, None, f"Error {resp.status_code}: {resp.text[:200]}"


# ── Registry ──────────────────────────────────────────────────────────────────

FETCHERS = {
    "OpenAI": fetch_models_openai,
    "Anthropic": fetch_models_anthropic,
    "Google Gemini": fetch_models_gemini,
    "Groq": fetch_models_groq,
    "Mistral": fetch_models_mistral,
    "Cohere": fetch_models_cohere,
    "Together AI": fetch_models_together,
    "Perplexity": fetch_models_perplexity,
    "HuggingFace": fetch_models_huggingface,
    "Sarvam AI": fetch_models_sarvam,
}

CALLERS = {
    "OpenAI": call_openai,
    "Anthropic": call_anthropic,
    "Google Gemini": call_gemini,
    "Groq": call_groq,
    "Mistral": call_mistral,
    "Cohere": call_cohere,
    "Together AI": call_together,
    "Perplexity": call_perplexity,
    "HuggingFace": call_huggingface,
    "Sarvam AI": call_sarvam,
}

INSTALL_HINTS = {
    "OpenAI": "pip install openai",
    "Anthropic": "pip install anthropic",
    "Google Gemini": "pip install google-generativeai",
    "Groq": "pip install groq",
    "Mistral": "pip install mistralai",
    "Cohere": "pip install cohere",
}

# ── Session state init ────────────────────────────────────────────────────────

if "fetched_provider" not in st.session_state:
    st.session_state.fetched_provider = None
if "fetched_models" not in st.session_state:
    st.session_state.fetched_models = []
if "fetch_error" not in st.session_state:
    st.session_state.fetch_error = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Configuration")

    AI_model = st.selectbox(
        "① Select AI Provider",
        list(FETCHERS.keys()),
        key="ai_provider",
    )

    API_key = st.text_input(
        "② Enter API Key",
        type="password",
        placeholder="Paste your key here...",
        key="api_key_input",
    )

    fetch_btn = st.button("🔍 Fetch Available Models", use_container_width=True)

    # Reset model list if provider or key changes
    if (st.session_state.fetched_provider != AI_model):
        st.session_state.fetched_models = []
        st.session_state.fetched_provider = None
        st.session_state.fetch_error = None

    # ── Step 1: Fetch models ──
    if fetch_btn:
        if not API_key.strip():
            st.warning("⚠️ Enter an API key first.")
        else:
            with st.spinner(f"Validating key & fetching {AI_model} models..."):
                try:
                    ok, models, err = FETCHERS[AI_model](API_key.strip())
                    if ok:
                        st.session_state.fetched_models = models
                        st.session_state.fetched_provider = AI_model
                        st.session_state.fetch_error = None
                    else:
                        st.session_state.fetched_models = []
                        st.session_state.fetch_error = err
                except ModuleNotFoundError as e:
                    hint = INSTALL_HINTS.get(AI_model, "")
                    st.session_state.fetch_error = f"Missing library: `{e.name}`. Run: `{hint}`"
                except Exception as e:
                    st.session_state.fetch_error = str(e)

    # Show fetch error inline
    if st.session_state.fetch_error:
        st.error(f"❌ {st.session_state.fetch_error}")

    # ── Step 2: Model picker (only after successful fetch) ──
    selected_model = None
    if st.session_state.fetched_models and st.session_state.fetched_provider == AI_model:
        st.success(f"✅ Key valid! {len(st.session_state.fetched_models)} models found.")
        selected_model = st.selectbox(
            "③ Select Model",
            st.session_state.fetched_models,
            key="model_picker",
        )
        test_btn = st.button("🚀 Test with this Model", use_container_width=True, type="primary")
    else:
        test_btn = False

    st.divider()
    st.caption(f"**Test prompt:**\n\n_{TEST_PROMPT}_")

# ── Main area ─────────────────────────────────────────────────────────────────

if not st.session_state.fetched_models:
    st.info("👈 Select a provider, enter your API key, and click **Fetch Available Models** to begin.")

elif test_btn and selected_model:
    with st.spinner(f"Calling **{selected_model}**..."):
        try:
            ok, response, err = CALLERS[AI_model](API_key.strip(), selected_model)
            if ok:
                st.success(f"✅ **{AI_model}** · `{selected_model}` is working!")
                st.divider()
                st.markdown("**📨 Prompt sent to AI:**")
                st.info(TEST_PROMPT)
                st.markdown("**🤖 AI Response:**")
                st.success(response)
            else:
                st.error(f"❌ Call failed: {err}")
        except ModuleNotFoundError as e:
            hint = INSTALL_HINTS.get(AI_model, "")
            st.error(f"❌ Missing library: `{e.name}`")
            if hint:
                st.code(hint, language="bash")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

elif st.session_state.fetched_models:
    col1, col2 = st.columns(2)
    col1.metric("Provider", AI_model)
    col2.metric("Models Available", len(st.session_state.fetched_models))
    with st.expander(f"📋 All {len(st.session_state.fetched_models)} available models", expanded=True):
        cols = st.columns(2)
        for i, m in enumerate(st.session_state.fetched_models):
            cols[i % 2].markdown(f"- `{m}`")
    st.info("👈 Pick a model from the sidebar and click **Test with this Model**.")