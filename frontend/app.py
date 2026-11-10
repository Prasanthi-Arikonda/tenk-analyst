"""Stage 8: the Streamlit page. It talks to the back end only through HTTP (POST /ask).

Start it with:  uv run python -m streamlit run frontend/app.py
"""
import os
import secrets
import uuid

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.environ.get("API_URL", "http://localhost:8000")

COMPANIES = {  # ticker -> name shown in the page
    "TSLA": "Tesla", "GM": "General Motors", "F": "Ford", "RIVN": "Rivian", "LCID": "Lucid",
    "APTV": "Aptiv", "BWA": "BorgWarner", "ALV": "Autoliv", "HOG": "Harley-Davidson", "CVNA": "Carvana",
}
YEARS = [2024, 2025]
PIPELINES = ["agent", "simple", "filters", "rerank", "decompose"]
EXAMPLES = [
    "What risks does Tesla report about battery supply?",
    "Compare GM and Ford R&D in 2025",
    "How did Carvana's revenue change from 2024 to 2025?",
]
WAKING = "The server is waking up. This takes about 30 seconds."

st.set_page_config(page_title="10-K Analyst", page_icon="📊")


# ---- Screen 1: the access code ----
def access_screen():
    st.title("10-K Analyst")
    expected = os.environ.get("ACCESS_CODE")
    if not expected:
        st.error("The ACCESS_CODE environment variable is not set.")
        st.stop()
    # A form lets Enter submit the code, the same as clicking the button.
    with st.form("access", clear_on_submit=True):  # clear the field after each try
        code = st.text_input("Access code", type="password")
        submitted = st.form_submit_button("Enter")
    if submitted:
        if secrets.compare_digest(code, expected):
            st.session_state.code = code
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong access code.")
    st.stop()


if not st.session_state.get("authed"):
    access_screen()

# ---- State that lasts for the whole browser session ----
if "chat_id" not in st.session_state:
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []

# ---- Sidebar ----
with st.sidebar:
    st.header("Filters")
    picked_companies = st.multiselect("Companies", list(COMPANIES), default=list(COMPANIES),
                                      format_func=lambda t: f"{COMPANIES[t]} ({t})")
    picked_years = st.multiselect("Years", YEARS, default=YEARS, format_func=lambda y: f"FY{y}")
    show_sources = st.checkbox("Show sources", value=True)
    if st.button("New chat"):
        st.session_state.chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    with st.expander("Learning mode"):
        pipeline = st.radio("Pipeline", PIPELINES, index=0,
                            help="agent has tools and memory. The others only search the 10-K text.")

# Everything selected means "no limit", so we send an empty list.
companies = [] if len(picked_companies) == len(COMPANIES) else picked_companies
years = [] if len(picked_years) == len(YEARS) else picked_years


# ---- Helpers ----
def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    v = abs(value)
    return f"{sign}${v / 1e9:.1f} billion" if v >= 0.05e9 else f"{sign}${v / 1e6:.0f} million"


def ask_api(question: str) -> dict:
    """Send the question to the back end. Try once more if it does not answer (it may be asleep)."""
    body = {"question": question, "chat_id": st.session_state.chat_id, "pipeline": pipeline,
            "companies": companies, "years": years}
    headers = {"X-Access-Code": st.session_state.code}
    for attempt in (1, 2):
        try:
            r = httpx.post(f"{API_URL}/ask", json=body, headers=headers, timeout=90)
            return r.json()
        except (httpx.TimeoutException, httpx.ConnectError):
            if attempt == 1:
                st.info(WAKING)
    return {"error": "The server did not answer. Try again in a minute."}


def show_extras(msg: dict):
    """Under an answer: the tool labels (always) and the sources (when 'Show sources' is on)."""
    if msg.get("tools_used"):
        st.caption("🔧 Tools used: " + ", ".join(msg["tools_used"]))
    if show_sources and msg.get("sources"):
        with st.expander(f"Sources ({len(msg['sources'])})"):
            for s in msg["sources"]:
                name = COMPANIES.get(s["company"], s["company"])
                if s["type"] == "sec":
                    st.markdown(f"**{name}, FY{s['year']}, SEC data** — {s['metric']} = {money(s['value'])} "
                                f"(tag `{s['tag']}`)")
                else:
                    st.markdown(f"**{name}, FY{s['year']} 10-K, Item {s['item']}** — {s.get('item_title', '')}")
                    st.caption(s["excerpt"] + " ...")
                    if s.get("url"):
                        st.markdown(f"[Open the filing on SEC.gov]({s['url']})")


# ---- The page ----
st.title("10-K Analyst")
st.caption("Questions about the 10-K filings of ten US auto companies, fiscal years 2024 and 2025.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            show_extras(msg)

question = None
examples_box = st.empty()  # a slot we can empty as soon as a question is asked
if not st.session_state.messages:
    with examples_box.container():
        st.write("Try an example:")
        for example in EXAMPLES:
            if st.button(example):
                question = example
question = st.chat_input("Ask about the 10-K filings") or question

if question:
    examples_box.empty()  # hide the example buttons while the answer is being made
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the filings..."):
            data = ask_api(question)
        if "error" in data:
            msg = {"role": "assistant", "content": data["error"], "tools_used": [], "sources": []}
        else:
            msg = {"role": "assistant", "content": data["answer"], "tools_used": data["tools_used"],
                   "sources": data["sources"]}
        st.markdown(msg["content"])
        show_extras(msg)
    st.session_state.messages.append(msg)
