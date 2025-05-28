# streamlit_app.py
# =============================================
# Web App de tradução de PDFs (offline) com Streamlit
# Autor: GPT/Gemini
# Data: 2025-05-28
# Versão: 2.0
# 
# Melhorias implementadas:
# 1. Spinner durante a tradução
# 2. Suporte a mais idiomas com seleção amigável
# 3. Preview do texto traduzido
# 4. Barra de progresso detalhada
# 5. Melhor tratamento de erros
# 6. Otimização de memória com generators
# 7. Configuração de threads para CPU
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
# CONFIGURAÇÕES GLOBAIS
# ------------------------------------------------

# Mapeamento de idiomas amigáveis
LANG_MAP = {
    "Inglês": "en",
    "Português": "pt",
    "Espanhol": "es",
    "Francês": "fr",
    "Alemão": "de",
    "Italiano": "it",
    "Holandês": "nl",
    "Russo": "ru",
    "Chinês (Simplificado)": "zh",
    "Japonês": "ja",
    "Coreano": "ko",
    "Árabe": "ar"
}

# Configuração de threads para CPU
TORCH_THREADS = 4

# ------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_translation_model(source_lang: str, target_lang: str):
    """
    Carrega tokenizer e modelo MarianMT para o par de idiomas informado.
    Usa HuggingFace (Helsinki-NLP/opus-mt-{src}-{tgt}). 
    Armazena em cache para não recarregar a cada tradução.
    """
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo de tradução: {str(e)}")
        st.stop()

def extract_text_from_pdf(filepath: str):
    """
    Extrai texto de cada página do PDF usando pdfplumber.
    Retorna lista de strings, uma por página.
    """
    try:
        pages = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                pages.append(txt)
        return pages
    except Exception as e:
        st.error(f"❌ Erro fatal ao ler PDF: {str(e)}")
        return []

def chunk_text(text: str, max_chars: int = 1500):
    """
    Divide um texto em chunks de até max_chars caracteres,
    tentando preservar parágrafos (quebras de linha dupla).
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    
    for p in paragraphs:
        # Remove espaços excessivos
        p = ' '.join(p.split())
        
        if len(p) > max_chars:
            # Se parágrafo é muito grande, quebra em frases
            sentences = p.split('. ')
            for sentence in sentences:
                if len(current) + len(sentence) < max_chars:
                    current += sentence + '. '
                else:
                    if current: chunks.append(current.strip())
                    current = sentence + '. '
        else:
            if len(current) + len(p) + 2 <= max_chars:
                current += p + "\n\n"
            else:
                if current: chunks.append(current.strip())
                current = p + "\n\n"
    
    if current:
        chunks.append(current.strip())
    
    return chunks

def translate_chunks(chunks, tokenizer, model, device):
    """
    Traduz cada chunk com o modelo MarianMT usando generator.
    Retorna lista de chunks traduzidos.
    """
    for chunk in chunks:
        try:
            inputs = tokenizer(chunk, 
                               return_tensors="pt", 
                               truncation=True, 
                               max_length=512,
                               padding="longest")
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated = model.generate(**inputs)
            
            text = tokenizer.decode(translated[0], skip_special_tokens=True)
            yield text
            
        except Exception as e:
            st.warning(f"⚠️ Erro na tradução de um bloco: {str(e)}")
            yield "[ERRO DE TRADUÇÃO]"

def reassemble_pages(translated_pages_chunks):
    """
    Recebe lista de páginas, cada página como lista de chunks traduzidos.
    Retorna lista de páginas completas (concatenando chunks).
    """
    pages_final = []
    for page_chunks in translated_pages_chunks:  # CORREÇÃO: "transled" -> "translated"
        page_text = "\n\n".join(page_chunks)
        pages_final.append(page_text)
    return pages_final

def create_pdf(bytesio_buffer, pages_text):
    """
    Gera um PDF simples em memória (BytesIO) a partir da lista pages_text.
    Cada página de texto vira uma página do PDF. 
    """
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        
        for page_text in pages_text:
            pdf.add_page()
            # Quebra texto em linhas de 100 caracteres
            lines = []
            for paragraph in page_text.split('\n'):
                while len(paragraph) > 0:
                    lines.append(paragraph[:100])
                    paragraph = paragraph[100:]
            
            for line in lines:
                pdf.cell(0, 10, txt=line, ln=True)
        
        pdf.output(bytesio_buffer)
        bytesio_buffer.seek(0)
        return bytesio_buffer
    except Exception as e:
        st.error(f"❌ Erro ao gerar PDF: {str(e)}")
        return None

def save_txt(pages_text):
    """
    Grava o texto traduzido em um BytesIO como arquivo .txt.
    """
    try:
        buf = BytesIO()
        full_text = "\f\n".join(pages_text)
        buf.write(full_text.encode("utf-8", errors="ignore"))
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"❌ Erro ao gerar TXT: {str(e)}")
        return None

def process_pdf_file(uploaded_file, source_lang, target_lang, output_format, device, progress_bar, status_text):
    """
    Processa um arquivo PDF completo com feedback de progresso
    """
    # 1) Salva o PDF em arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    base_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]

    # 2) Extrai páginas
    status_text.text(f"📖 Lendo {uploaded_file.name}...")
    pages = extract_text_from_pdf(tmp_path)
    
    if not pages:
        os.remove(tmp_path)
        return {"nome_base": base_name, "pdf_buffer": None, "txt_buffer": None, "pages": []}

    # 3) Carrega modelo e tokenizer
    status_text.text(f"⚙️ Carregando modelo de tradução...")
    tokenizer, model = load_translation_model(source_lang, target_lang)
    model = model.to(device)
    model.eval()

    # 4) Para cada página, divide em chunks e traduz
    translated_pages_chunks = []
    total_pages = len(pages)
    
    for i, page_text in enumerate(pages, start=1):
        status_text.text(f"🌐 Traduzindo página {i}/{total_pages}...")
        progress_bar.progress(i / total_pages)
        
        chunks = chunk_text(page_text)
        translated_chunks = list(translate_chunks(chunks, tokenizer, model, device))
        translated_pages_chunks.append(translated_chunks)

    # 5) Remonta páginas completas
    translated_pages = reassemble_pages(translated_pages_chunks)

    pdf_buf = None
    txt_buf = None

    # 6) Gera saída(s) conforme formato solicitado
    if output_format in ("pdf", "both"):
        status_text.text(f"📄 Gerando PDF traduzido...")
        buf = BytesIO()
        pdf_buf = create_pdf(buf, translated_pages)

    if output_format in ("txt", "both"):
        status_text.text(f"📝 Gerando TXT traduzido...")
        txt_buf = save_txt(translated_pages)

    # 7) Limpa arquivo temporário
    os.remove(tmp_path)

    return {
        "nome_base": base_name,
        "pdf_buffer": pdf_buf,
        "txt_buffer": txt_buf,
        "pages": translated_pages
    }


# ------------------------------------------------
# STREAMLIT APP
# ------------------------------------------------

st.set_page_config(page_title="PDF Translator", layout="wide")
st.title("🌍 PDF Translator (Offline)")

st.markdown(
    """
    <style>
    .small-font { font-size:0.9rem !important; }
    .warning { background-color: #fff4f4; border-radius: 5px; padding: 10px; }
    </style>
    
    Traduza PDFs para mais de 10 idiomas offline usando inteligência artificial. 
    **Funciona 100% localmente** - seus documentos nunca saem do seu computador.
    """, 
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.header("⚙️ Configurações")

# Seletores de idioma
source_options = list(LANG_MAP.keys())
target_options = list(LANG_MAP.keys())

col1, col2 = st.sidebar.columns(2)
with col1:
    source_lang_name = st.selectbox("Idioma de origem", options=source_options, index=1)
with col2:
    target_lang_name = st.selectbox("Idioma de destino", options=target_options, index=0)

source_lang = LANG_MAP[source_lang_name]
target_lang = LANG_MAP[target_lang_name]

# Outras opções
output_format = st.sidebar.selectbox("Formato de saída", options=["pdf", "txt", "both"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Otimizações**")
split_size = st.sidebar.slider("Tamanho dos blocos de tradução", 500, 3000, 1500, 100)

# Upload de arquivos
st.subheader("📤 Envio de Documentos")
uploaded_files = st.file_uploader(
    "Selecione PDFs para traduzir", 
    type=["pdf"], 
    accept_multiple_files=True,
    help="Máx. 5 arquivos por vez para melhor desempenho"
)

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} arquivo(s) pronto(s) para tradução")

# Detecção automática de GPU/CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(TORCH_THREADS)
    st.sidebar.info(f"Usando CPU ({TORCH_THREADS} threads)")
else:
    st.sidebar.success(f"GPU detectada! ({torch.cuda.get_device_name(0)})")

# Botão de tradução
if st.button("🚀 Iniciar Tradução", type="primary", disabled=not uploaded_files):
    if not uploaded_files:
        st.error("Por favor, envie pelo menos um arquivo PDF")
    else:
        # Contêiner para status
        status_container = st.empty()
        progress_bar = st.progress(0)
        download_section = st.container()
        
        resultados = []
        for i, uploaded_file in enumerate(uploaded_files):
            # Atualiza status
            status_container.subheader(f"📂 Processando: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            status_text = status_container.empty()
            
            # Processa o arquivo
            with st.spinner(f"Traduzindo {uploaded_file.name}..."):
                res = process_pdf_file(
                    uploaded_file,
                    source_lang,
                    target_lang,
                    output_format,
                    device,
                    progress_bar,
                    status_text
                )
                resultados.append(res)
            
            # Reset da barra de progresso para próximo arquivo
            progress_bar.progress(0)
        
        # Finalização
        status_container.success("✅ Tradução concluída com sucesso!")
        st.balloons()
        
        # Seção de download
        download_section.header("📥 Downloads")
        
        for res in resultados:
            base = res["nome_base"]
            
            if res["pages"]:
                with st.expander(f"📄 Pré-visualização: {base}"):
                    st.text_area("Texto traduzido (primeira página)", 
                                value=res["pages"][0][:2000] + ("..." if len(res["pages"][0]) > 2000 else ""), 
                                height=300)
            
            col1, col2 = download_section.columns(2)
            
            if res["pdf_buffer"] and output_format in ("pdf", "both"):
                with col1:
                    st.download_button(
                        label=f"⬇️ Baixar PDF: {base}_translated.pdf",
                        data=res["pdf_buffer"].getvalue(),
                        file_name=f"{base}_translated.pdf",
                        mime="application/pdf",
                    )
            
            if res["txt_buffer"] and output_format in ("txt", "both"):
                with col2:
                    st.download_button(
                        label=f"⬇️ Baixar TXT: {base}_translated.txt",
                        data=res["txt_buffer"].getvalue(),
                        file_name=f"{base}_translated.txt",
                        mime="text/plain",
                    )

# Informações de rodapé
st.markdown("---")
st.markdown("""
    ### 💡 Como usar:
    1. Selecione os idiomas de origem e destino
    2. Envie um ou mais arquivos PDF
    3. Clique em "Iniciar Tradução"
    4. Baixe os resultados quando estiverem prontos

    ### ⚠️ Limitações:
    - Documentos muito grandes podem demorar vários minutos
    - Formatação complexa pode não ser preservada
    - Traduções literais podem perder nuances
    - Máquinas sem GPU serão mais lentas

    **Dica:** Para documentos grandes, reduza o tamanho dos blocos na sidebar
""")

st.caption("🛠️ Desenvolvido com Streamlit, PyTorch e MarianMT | v2.0")
