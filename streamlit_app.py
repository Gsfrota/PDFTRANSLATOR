# streamlit_app.py
# =============================================
# Web App de tradução de PDFs (offline) com Streamlit
# Autor: GPT/Gemini
# Data: 2025-05-28
# Versão: 1.0
#
# - Extrai texto de PDF(s) usando pdfplumber.
# - Tradução offline com MarianMT (HuggingFace).
# - Gera PDF e/ou TXT traduzido.
# - Interface simples em Streamlit: upload de arquivos, escolha de idiomas e botão “Translate”.
#
# Dependências:
#   pip install streamlit pdfplumber transformers torch sentencepiece fpdf2
# =============================================

import streamlit as st
import pdfplumber
from fpdf import FPDF
from transformers import MarianMTModel, MarianTokenizer
import torch
import tempfile
import os
from io import BytesIO

# ------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_translation_model(source_lang: str, target_lang: str):
    """
    Carrega tokenizer e modelo MarianMT para o par de idiomas informado.
    Usa HuggingFace (Helsinki-NLP/opus-mt-{src}-{tgt}). 
    Armazena em cache para não recarregar a cada traduÇÃO.
    """
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def extract_text_from_pdf(filepath: str):
    """
    Extrai texto de cada página do PDF usando pdfplumber.
    Retorna lista de strings, uma por página.
    """
    pages = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(txt)
    except Exception as e:
        st.warning(f"❗️ Erro ao ler PDF {os.path.basename(filepath)}: {e}")
    return pages

def chunk_text(text: str, max_chars: int = 4000):
    """
    Divide um texto em chunks de até max_chars caracteres,
    tentando preservar parágrafos (quebras de linha dupla).
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for p in paragraphs:
        if len(p) > max_chars:
            # se parágrafo é muito grande, quebra pelo tamanho fixo
            if current:
                chunks.append(current.strip())
                current = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i : i + max_chars].strip())
        else:
            if len(current) + len(p) + 2 <= max_chars:
                current += p + "\n\n"
            else:
                chunks.append(current.strip())
                current = p + "\n\n"
    if current:
        chunks.append(current.strip())
    return chunks

def translate_chunks(chunks, tokenizer, model, device):
    """
    Traduz cada chunk com o modelo MarianMT. Retorna lista de chunks traduzidos.
    """
    translated_chunks = []
    for chunk in chunks:
        # preparar inputs
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding="longest")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # gerar tradução
        with torch.no_grad():
            translated = model.generate(**inputs)
        text = tokenizer.decode(translated[0], skip_special_tokens=True)
        translated_chunks.append(text)
    return translated_chunks

def reassemble_pages(translated_pages_chunks):
    """
    Recebe lista de páginas, cada página como lista de chunks traduzidos.
    Retorna lista de páginas completas (concatenando chunks).
    """
    pages_final = []
    for page_chunks in translated_pages_chunks:
        page_text = "\n\n".join(page_chunks)
        pages_final.append(page_text)
    return pages_final

def create_pdf(bytesio_buffer, pages_text):
    """
    Gera um PDF simples em memória (BytesIO) a partir da lista pages_text.
    Cada página de texto vira uma página do PDF. 
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for i, page_text in enumerate(pages_text):
        pdf.add_page()
        for line in page_text.split("\n"):
            pdf.multi_cell(0, 10, line)
    pdf.output(bytesio_buffer)
    bytesio_buffer.seek(0)
    return bytesio_buffer

def save_txt(bytesio_buffer, pages_text):
    """
    Grava o texto traduzido em um BytesIO como arquivo .txt, cada página separada por \f.
    """
    buf = BytesIO()
    for i, page_text in enumerate(pages_text):
        buf.write(page_text.encode("utf-8", errors="ignore"))
        if i < len(pages_text) - 1:
            buf.write(b"\n\f\n")
    buf.seek(0)
    return buf

def process_pdf_file(uploaded_file, source_lang, target_lang, output_format, device):
    """
    Recebe um arquivo PDF (UploadedFile do Streamlit), executa:
      - salva em temporário,
      - extrai texto,
      - divide em chunks,
      - traduz,
      - monta PDF/TXT traduzido em memória. 
    Retorna dicionário: {
       "nome_base": ..., 
       "pdf_buffer": BytesIO (ou None),
       "txt_buffer": BytesIO (ou None)
    }
    """
    # 1) Salva o PDF em arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    base_name = os.path.splitext(os.path.basename(tmp_path))[0]

    # 2) Extrai páginas
    pages = extract_text_from_pdf(tmp_path)
    if not pages:
        return { "nome_base": base_name, "pdf_buffer": None, "txt_buffer": None }

    # 3) Carrega modelo e tokenizer (em cache)
    tokenizer, model = load_translation_model(source_lang, target_lang)
    model = model.to(device)

    # 4) Para cada página, divide em chunks e traduz
    translated_pages_chunks = []
    for i, page_text in enumerate(pages, start=1):
        st.write(f"Translating page {i} of {len(pages)} for file {base_name}.pdf...")
        chunks = chunk_text(page_text)
        translated_chunks = translate_chunks(chunks, tokenizer, model, device)
        translated_pages_chunks.append(translated_chunks)

    # 5) Reassemble páginas completas
    translated_pages = reassemble_pages(translated_pages_chunks)

    pdf_buf = None
    txt_buf = None

    # 6) Gera saída(s) conforme formato solicitado
    if output_format in ("pdf", "both"):
        buf = BytesIO()
        buf = create_pdf(buf, translated_pages)
        pdf_buf = buf

    if output_format in ("txt", "both"):
        buf2 = save_txt(BytesIO(), translated_pages)
        txt_buf = buf2

    # 7) Limpa arquivo temporário
    os.remove(tmp_path)

    return { "nome_base": base_name, "pdf_buffer": pdf_buf, "txt_buffer": txt_buf }


# ------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------

st.set_page_config(page_title="PDF Translator", layout="wide")
st.title("🌐 PDF Translator (Offline)")

st.markdown(
    """
    Este app permite que você envie um ou vários arquivos PDF, escolha idioma de origem e destino, e receba de volta:
    - PDF totalmente traduzido, mantendo quebras de página, ou  
    - Arquivo TXT com o conteúdo traduzido.  
    **Tudo 100% offline**, sem custos de API externa, usando modelos MarianMT (HuggingFace).
    """
)

st.sidebar.header("Configurações")
source_lang = st.sidebar.text_input("Idioma de origem (ISO 639-1)", value="pt")
target_lang = st.sidebar.text_input("Idioma de destino (ISO 639-1)", value="en")
output_format = st.sidebar.selectbox("Formato de saída", options=["pdf", "txt", "both"])

uploaded_files = st.file_uploader(
    "Selecione um ou vários PDFs para traduzir", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"{len(uploaded_files)} arquivo(s) selecionado(s). Pronto para traduzir.")
else:
    st.warning("Envie ao menos um arquivo PDF para habilitar a tradução.")

if st.button("🚀 Traduzir todos os PDFs"):
    if not uploaded_files:
        st.error("Nenhum arquivo PDF enviado. Selecione ao menos um antes de clicar em Traduzir.")
    else:
        # Detectar se GPU está disponível para usar no modelo (caso contrário, CPU)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.write(f"Using device: {device}")

        resultados = []
        for uploaded_file in uploaded_files:
            st.write(f"---\n### Processando: {uploaded_file.name}")
            res = process_pdf_file(uploaded_file, source_lang, target_lang, output_format, device)
            resultados.append(res)

        st.success("🎉 Tradução concluída para todos os arquivos!")

        st.markdown("---")
        st.header("📥 Downloads")

        for res in resultados:
            base = res["nome_base"]
            st.subheader(f"Arquivo: {base}.pdf")
            if res["pdf_buffer"] is not None and output_format in ("pdf", "both"):
                st.download_button(
                    label=f"Baixar {base}_translated.pdf",
                    data=res["pdf_buffer"].getvalue(),
                    file_name=f"{base}_translated.pdf",
                    mime="application/pdf",
                )
            if res["txt_buffer"] is not None and output_format in ("txt", "both"):
                st.download_button(
                    label=f"Baixar {base}_translated.txt",
                    data=res["txt_buffer"].getvalue(),
                    file_name=f"{base}_translated.txt",
                    mime="text/plain",
                )

st.markdown("---")
st.markdown(
    """
    **Instruções rápidas para rodar localmente**:

    1. Instale dependências:
       ```
       pip install streamlit pdfplumber transformers torch sentencepiece fpdf2
       ```
    2. Baixe (ou copie) este arquivo como `streamlit_app.py`.
    3. No terminal, execute:
       ```
       streamlit run streamlit_app.py
       ```
    4. Abra a URL indicada (normalmente http://localhost:8501) no navegador.
    5. Use a interface para enviar seus PDFs, escolher idiomas e clicar em “Traduzir”.  
    6. Baixe os arquivos traduzidos ao final.

    **Deploy no Streamlit Cloud**:

    1. Crie um repositório no GitHub contendo apenas este `streamlit_app.py`.
    2. Acesse https://streamlit.io/cloud e conecte seu GitHub.
    3. Escolha o repositório, pressione “Deploy”.  
    4. Em segundos seu app estará disponível publicamente; qualquer pessoa pode enviar PDFs e baixar traduções.

    Pronto! Sem precisar configurar servidor nem lidar com Google Cloud ou APIs pagas.  
    """
)


