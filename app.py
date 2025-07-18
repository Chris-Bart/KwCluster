# %%
import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Groq API Key aus secrets laden
if "groq" in st.secrets and "api_key" in st.secrets["groq"]:
    os.environ["GROQ_API_KEY"] = st.secrets["groq"]["api_key"]
else:
    st.error("Groq API Key nicht gefunden. Bitte in .streamlit/secrets.toml einfügen.")
    st.stop()

def clean_google_sheet_url(url: str) -> str:
    """Wandelt edit-URLs in export-CSV-URLs um"""
    if "edit#gid=" in url:
        return url.replace("/edit#gid=", "/export?format=csv&gid=")
    elif "spreadsheets/d/" in url and "gid=" in url and "export" not in url:
        # Falls jemand z. B. händisch was mit gid gebastelt hat
        doc_id = url.split("/d/")[1].split("/")[0]
        gid = url.split("gid=")[-1]
        return f"https://docs.google.com/spreadsheets/d/{doc_id}/export?format=csv&gid={gid}"
    return url


# Streamlit UI
st.title("Keyword Clustering mit LLM")
sheet_url = st.text_input("🔗 Google Sheets URL (öffentlich freigegeben):")

if sheet_url:
    cleaned_url = clean_google_sheet_url(sheet_url)
    st.caption(f"📎 Verwendete CSV-URL: {cleaned_url}")


# Funktion: Sheet laden
def load_keywords_from_sheet(url):
    try:
        df = pd.read_csv(cleaned_url, header=None, skiprows=3, usecols=[0])

        MAX_KEYWORDS = 500
        keywords = df[0].dropna().astype(str).tolist()[:MAX_KEYWORDS]

        return keywords
    except Exception as e:
        st.error(f"Fehler beim Laden des Sheets: {e}")
        return None

# LLM vorbereiten
model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein KI-Modell zur Analyse von Marketing-Keywords. Deine Aufgabe ist es, ähnliche Keywords in sinnvolle Kategorien zu clustern."),
    ("user", "Cluster die folgenden Suchbegriffe in sinnvolle Kategorien und gib die Cluster in klarer Struktur zurück:\n\n{keywords}")
])
chain = prompt_template | model | StrOutputParser()

# Button für Analyse
if st.button("Analyse starten") and sheet_url:
    with st.spinner("Lade Daten und analysiere..."):
        keywords = load_keywords_from_sheet(sheet_url)
        if keywords:
            keyword_text = "\n".join(keywords)
            result = chain.invoke({"keywords": keyword_text})
            st.subheader("🧾 Ergebnis:")
            st.text(result)
