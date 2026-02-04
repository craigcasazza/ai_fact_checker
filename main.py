import streamlit as st
from google import genai
from google.genai import types
import os
import json
import re
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="DJ Editorial Assistant", page_icon="📰", layout="wide")

# --- CSS: Highlighting, Cards, & Metrics ---
st.markdown("""
    <style>
    /* Fact Cards */
    .fact-card { padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #eee; border-left-width: 5px; background-color: white; }
    
    /* Highlight Colors */
    .highlight-TRUE { background-color: #d1e7dd; color: #0f5132; padding: 2px 4px; border-radius: 4px; text-decoration: none; border-bottom: 2px solid #28a745; cursor: pointer; }
    .highlight-FALSE { background-color: #f8d7da; color: #842029; padding: 2px 4px; border-radius: 4px; text-decoration: none; border-bottom: 2px solid #dc3545; cursor: pointer; }
    .highlight-PARTIAL { background-color: #fff3cd; color: #664d03; padding: 2px 4px; border-radius: 4px; text-decoration: none; border-bottom: 2px solid #ffc107; cursor: pointer; }
    .highlight-UNCERTAIN { background-color: #e2e3e5; color: #41464b; padding: 2px 4px; border-radius: 4px; text-decoration: none; border-bottom: 2px solid #6c757d; cursor: pointer; }
    
    /* Status Colors */
    .status-TRUE { border-left-color: #28a745; }
    .status-FALSE { border-left-color: #dc3545; }
    .status-PARTIAL { border-left-color: #ffc107; }
    .status-UNCERTAIN { border-left-color: #6c757d; }

    /* Metric Scorecards */
    .metric-box { text-align: center; padding: 10px; border-radius: 5px; background-color: #f8f9fa; border: 1px solid #dee2e6; margin-bottom: 10px; }
    .metric-score { font-size: 24px; font-weight: bold; color: #333; }
    .metric-label { font-size: 14px; color: #666; }
    
    /* Rewrite Box */
    .rewrite-box { background-color: #f0f7ff; padding: 20px; border-radius: 8px; border-left: 5px solid #007bff; margin-top: 20px; margin-bottom: 20px; }
    
    /* Hover effects */
    a[class^="highlight-"]:hover { opacity: 0.8; }
    </style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'results' not in st.session_state: st.session_state.results = None
if 'eval_scores' not in st.session_state: st.session_state.eval_scores = None
if 'rewritten_text' not in st.session_state: st.session_state.rewritten_text = None
if 'raw_text' not in st.session_state: st.session_state.raw_text = ""

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("Gemini API Key", type="password")
    if api_key_input: os.environ["GEMINI_API_KEY"] = api_key_input

def get_client():
    if not os.environ.get("GEMINI_API_KEY"): return None
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# --- 1. Fact Decomposition (Accuracy) ---
def decompose_text(client, text):
    """Extracts claims and their exact quotes."""
    prompt = f"""
    Analyze the text. Identify distinct factual claims.
    For each claim, provide:
    1. "claim": The standalone statement.
    2. "quote": The EXACT substring from the text.
    
    Return JSON list: [{{"claim": "...", "quote": "..."}}]
    Text: {text}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except: return []

# --- 2. Fact Verification (Accuracy) ---
def verify_fact(client, fact_dict):
    """Verifies a claim using Google Search."""
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    prompt = f"""
    Verify this claim using Google Search.
    Claim: "{fact_dict.get('claim')}"
    Context: "{fact_dict.get('quote')}"
    Response format: VERDICT: [TRUE / FALSE / PARTIALLY TRUE / UNCERTAIN] REASONING: [Text]
    """
    try:
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(tools=[grounding_tool], temperature=0.0)
        )
    except: return None

# --- 3. Qualitative Evaluation (Relevance & Clarity) ---
def evaluate_quality(client, text, verification_summary):
    prompt = f"""
    You are a Dow Jones Editor. Evaluate based on these criteria.
    
    1. RELEVANCE (1-3 Scale):
       - Directness? Saliency? Disambiguation?
       - 3: Excellent. 2: Acceptable. 1: Poor.

    2. CLARITY (1-3 Scale):
       - Concise? Logical flow? Professional/Objective Tone?
       - 3: Excellent. 2: Acceptable. 1: Poor.

    INPUT TEXT: "{text}"
    FACT CHECK CONTEXT: {verification_summary}

    Return JSON: {{ "relevance_score": int, "relevance_reasoning": "...", "clarity_score": int, "clarity_reasoning": "..." }}
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        return {"relevance_score": 0, "relevance_reasoning": "Error", "clarity_score": 0, "clarity_reasoning": "Error"}

# --- 4. Auto-Rewrite (Optimization) ---
def rewrite_content(client, original_text, results, eval_scores):
    corrections = [f"Claim '{r['claim']}' was {r['verdict']}. Info: {r['reasoning']}" for r in results if r['verdict'] != 'TRUE']
    correction_prompt = "\n".join(corrections)
    
    prompt = f"""
    Rewrite the text to achieve a PERFECT score (Accuracy 100%, Relevance 3/3, Clarity 3/3).
    
    Fix these factual errors: {correction_prompt}
    Address these critiques:
    - Relevance: {eval_scores['relevance_reasoning']}
    - Clarity: {eval_scores['clarity_reasoning']}
    
    Maintain Dow Jones professional tone. Output ONLY the rewritten text.
    Original Text: "{original_text}"
    """
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text

# --- Export Helpers ---
def convert_to_csv(results):
    flat_results = []
    for r in results:
        source_str = " | ".join([s['url'] for s in r.get('sources', [])])
        flat_results.append({
            "Fact Claim": r.get('claim', ''), "Original Quote": r.get('quote', ''),
            "Verdict": r.get('verdict', ''), "Reasoning": r.get('reasoning', ''), "Sources": source_str
        })
    return pd.DataFrame(flat_results).to_csv(index=False).encode('utf-8')

def convert_to_markdown(results, scores, rewrite):
    md = "# Editorial Report\n\n"
    md += f"**Relevance Score:** {scores['relevance_score']}/3\n"
    md += f"**Clarity Score:** {scores['clarity_score']}/3\n\n"
    if rewrite:
        md += "## Optimized Rewrite\n" + rewrite + "\n\n"
    md += "## Fact Check Details\n"
    for r in results:
        icon = "✅" if r['verdict'] == "TRUE" else "❌" if r['verdict'] == "FALSE" else "⚠️"
        md += f"### {icon} {r.get('claim', '')}\n"
        md += f"**Verdict:** {r.get('verdict', '')}\n\n**Reasoning:** {r.get('reasoning', '')}\n\n---\n"
    return md

# --- Main Logic ---

st.title("📰 DJ Editorial Assistant")
user_text = st.text_area("Input Text", height=150, placeholder="Enter text...", value=st.session_state.raw_text)

if st.button("Evaluate & Optimize", type="primary"):
    client = get_client()
    st.session_state.raw_text = user_text
    st.session_state.results = None 
    
    if not client: st.error("Missing API Key"); st.stop()
    if not user_text: st.warning("Enter text"); st.stop()

    with st.status("Running Editorial Review...", expanded=True) as status:
        # 1. Decompose
        status.write("🔍 Extracting claims...")
        items = decompose_text(client, user_text)
        if not items: st.error("No claims found."); st.stop()
        
        # 2. Verify
        results = []
        progress = st.progress(0)
        for i, item in enumerate(items):
            status.write(f"🌍 Verifying: {item.get('claim', '')[:30]}...")
            resp = verify_fact(client, item)
            verdict, reasoning, sources = "UNCERTAIN", "", []
            if resp:
                txt = resp.text
                if "VERDICT: TRUE" in txt: verdict = "TRUE"
                elif "VERDICT: FALSE" in txt: verdict = "FALSE"
                elif "VERDICT: PARTIALLY" in txt: verdict = "PARTIALLY"
                reasoning = txt.split("REASONING:")[-1].strip() if "REASONING:" in txt else txt
                if resp.candidates[0].grounding_metadata.grounding_chunks:
                    sources = [{"title": c.web.title, "url": c.web.uri} for c in resp.candidates[0].grounding_metadata.grounding_chunks if c.web]
            
            results.append({"id": i, "claim": item.get('claim'), "quote": item.get('quote'), "verdict": verdict, "reasoning": reasoning, "sources": sources})
            progress.progress((i+1)/len(items))
        
        st.session_state.results = results
        
        # 3. Quality Check
        status.write("⚖️ Judging Relevance & Clarity...")
        acc_score = int((sum(1 for r in results if r['verdict'] == "TRUE") / len(results)) * 100)
        scores = evaluate_quality(client, user_text, f"Accuracy: {acc_score}%")
        st.session_state.eval_scores = scores
        
        # 4. Rewrite
        if acc_score < 100 or scores['relevance_score'] < 3 or scores['clarity_score'] < 3:
            status.write("✍️ Generating Optimized Rewrite...")
            st.session_state.rewritten_text = rewrite_content(client, user_text, results, scores)
        else:
            st.session_state.rewritten_text = None
        
        status.update(label="Complete", state="complete", expanded=False)

# --- Results UI ---

if st.session_state.results and st.session_state.eval_scores:
    scores = st.session_state.eval_scores
    acc_score = int((sum(1 for r in st.session_state.results if r['verdict'] == "TRUE") / len(st.session_state.results)) * 100)
    
    st.divider()
    
    # --- Metrics & Export ---
    c1, c2, c3, c4 = st.columns([1,1,1,1.5])
    
    c1.markdown(f"<div class='metric-box' style='border-top: 5px solid {'#28a745' if acc_score==100 else '#dc3545'}'>"
                f"<div class='metric-label'>Accuracy</div><div class='metric-score'>{acc_score}%</div></div>", unsafe_allow_html=True)
    
    c2.markdown(f"<div class='metric-box' style='border-top: 5px solid {'#28a745' if scores['relevance_score']==3 else '#ffc107'}'>"
                f"<div class='metric-label'>Relevance</div><div class='metric-score'>{scores['relevance_score']}/3</div></div>", unsafe_allow_html=True)
    
    c3.markdown(f"<div class='metric-box' style='border-top: 5px solid {'#28a745' if scores['clarity_score']==3 else '#ffc107'}'>"
                f"<div class='metric-label'>Clarity</div><div class='metric-score'>{scores['clarity_score']}/3</div></div>", unsafe_allow_html=True)
    
    with c4:
        st.caption("📤 **Export Report**")
        xc1, xc2 = st.columns(2)
        csv_data = convert_to_csv(st.session_state.results)
        xc1.download_button("CSV", csv_data, "report.csv", "text/csv")
        md_data = convert_to_markdown(st.session_state.results, scores, st.session_state.rewritten_text)
        xc2.download_button("Text", md_data, "report.md", "text/markdown")

    # --- Rewrite Section ---
    if st.session_state.rewritten_text:
        st.markdown(f"""
        <div class="rewrite-box">
            <h3 style="margin-top:0">✨ Optimized Version</h3>
            <p>{st.session_state.rewritten_text}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Interactive Highlighter ---
    st.subheader("Interactive Analysis")
    st.caption("Click highlighted text to view details.")
    
    highlighted_html = st.session_state.raw_text
    sorted_res = sorted(st.session_state.results, key=lambda x: len(x.get('quote', '')), reverse=True)
    
    for res in sorted_res:
        quote = res.get('quote', '')
        if quote:
            replacement = f'<a href="#fact-{res["id"]}" class="highlight-{res["verdict"]}" title="{res["claim"]}">{quote}</a>'
            try: highlighted_html = re.sub(re.escape(quote), replacement, highlighted_html, count=1)
            except: pass

    st.markdown(f"<div style='padding:20px; background:white; border:1px solid #ddd; line-height:1.6; font-size:1.1em;'>{highlighted_html}</div>", unsafe_allow_html=True)

    # --- Details Tabs ---
    st.divider()
    t1, t2 = st.tabs(["Fact Checks", "Editorial Critique"])
    
    with t1:
        for res in st.session_state.results:
            st.markdown(f"<div id='fact-{res['id']}'></div>", unsafe_allow_html=True)
            icon = "✅" if res['verdict'] == "TRUE" else "❌" if res['verdict'] == "FALSE" else "⚠️"
            with st.expander(f"{icon} {res.get('claim')}", expanded=(res['verdict']!='TRUE')):
                st.markdown(f"**Verdict:** {res['verdict']}\n\n**Reasoning:** {res['reasoning']}")
                if res['sources']:
                    st.markdown("**Sources:**")
                    for s in res['sources']: st.markdown(f"- [{s['title']}]({s['url']})")
    
    with t2:
        st.info(f"**Relevance Analysis:** {scores['relevance_reasoning']}")
        st.info(f"**Clarity Analysis:** {scores['clarity_reasoning']}")