import io
import pandas as pd
import streamlit as st
from openai import OpenAI

# -----------------
# Streamlit config
# -----------------
st.set_page_config(layout="wide")
st.title("Chat with your Excel File")
st.markdown("Upload an Excel file, specify the sheet you want to chat with, provide context if you'd like, and ask questions about the data.")

# -----------------
# Sidebar: API key & model
# -----------------
st.sidebar.header("LLM Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Only used locally in this session.")
model = st.sidebar.selectbox(
    "Choose model",
    ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
    index=0
)

# -----------------
# Sidebar: Optional context
# -----------------
st.sidebar.markdown("---")
st.sidebar.header("Sheet Context (Optional)")
sheet_context = st.sidebar.text_area(
    "Provide context about this data",
    placeholder="e.g., This dataset contains deposition excerpts, where each row includes a page/line reference, a summary of the testimony...",
    height=100,
    help="Optional: Describe what this data represents to improve format recommendation accuracy."
)

if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to proceed.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------
# File uploader
# -----------------
uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])
if uploaded is None:
    st.info("Waiting for a file…")
    st.stop()

# -----------------
# Process uploaded file
# -----------------
file_bytes = uploaded.read()
buf = io.BytesIO(file_bytes)

# Read Excel file
try:
    xls = pd.ExcelFile(buf, engine="openpyxl")
    sheet = st.selectbox("Select a sheet", options=xls.sheet_names)
    buf.seek(0)
    df = pd.read_excel(buf, sheet_name=sheet, engine="openpyxl")
    
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

# -----------------
# Analyze data and determine optimal format
# -----------------

# Create sample for LLM analysis (limit to prevent token overflow)
sample_size = min(15, len(df))
sample_df = df.head(sample_size)
sample_csv = sample_df.to_csv(index=False)

# Get basic data characteristics
file_size_mb = len(file_bytes) / (1024 * 1024)
num_numeric_cols = len(df.select_dtypes(include=['number']).columns)
num_text_cols = len(df.select_dtypes(include=['object']).columns)
total_cols = len(df.columns)

# Create analysis prompt
context_section = f"\n\nUser-provided context: {sheet_context}" if sheet_context.strip() else ""

analysis_prompt = f"""
Analyze this dataset sample and recommend the optimal format (CSV or Markdown) for LLM processing.

Dataset characteristics:
- File size: {file_size_mb:.2f} MB
- Rows: {len(df)}
- Columns: {total_cols}
- Numeric columns: {num_numeric_cols}
- Text columns: {num_text_cols}{context_section}

Sample data (first {sample_size} rows):
{sample_csv}

Consider:
1. Data structure complexity (simple tabular vs. complex relationships)
2. Qualitative vs quantitative balance
3. File size and parsing efficiency
4. How well the data would be preserved in each format
5. The user context (if provided) to understand data usage patterns

Respond with ONLY one word: either "CSV" or "Markdown"
"""

# Get format recommendation
with st.spinner("Analyzing data structure and determining optimal format..."):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data format expert. Respond with only 'CSV' or 'Markdown'."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1
        )
        recommended_format = response.choices[0].message.content.strip().upper()
        
        # Ensure valid response
        if recommended_format not in ["CSV", "MARKDOWN"]:
            recommended_format = "CSV"  # Default fallback
            
    except Exception as e:
        st.error(f"Error getting format recommendation: {e}")
        recommended_format = "CSV"  # Default fallback

st.success(f"Recommended format: **{recommended_format}** (This is just a transparency message and would be removed for production)")

# -----------------
# Convert and provide download
# -----------------
if recommended_format == "CSV":
    # Convert to CSV
    csv_output = df.to_csv(index=False)
    file_extension = "csv"
    mime_type = "text/csv"
    converted_content = csv_output
    
elif recommended_format == "MARKDOWN":
    # Convert to Markdown table
    md_output = df.to_markdown(index=False)
    file_extension = "md"
    mime_type = "text/markdown"
    converted_content = md_output

# Create download button
original_filename = uploaded.name.rsplit('.', 1)[0]  # Remove .xlsx extension
download_filename = f"{original_filename}_converted.{file_extension}"

st.download_button(
    label=f"Download {recommended_format} file (Also just for transparency)",
    data=converted_content,
    file_name=download_filename,
    mime=mime_type,
    help=f"Download the converted {recommended_format} file optimized for LLM processing"
)

# -----------------
# Show preview of converted format
# -----------------
st.markdown("---")
st.subheader("Preview of Data")

if recommended_format == "CSV":
    preview_df = df.head(10)
    st.dataframe(preview_df)
    
elif recommended_format == "MARKDOWN":
    preview_md = df.head(10).to_markdown(index=False)
    st.code(preview_md, language="markdown")

# -----------------
# User query section
# -----------------
st.markdown("---")
st.subheader("Chat with Your Data")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.chat_history):
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])

# Chat input
user_input = st.chat_input("Ask a question about your data...")

if user_input and converted_content:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.spinner("Analyzing..."):
        # Prepare the context section if provided
        context_section = f"\n\nContext about this dataset: {sheet_context}" if sheet_context.strip() else ""
        
        # Build conversation history for context
        conversation_context = ""
        if len(st.session_state.chat_history) > 1:
            conversation_context = "\n\nPrevious conversation:\n"
            for msg in st.session_state.chat_history[:-1]:  # Exclude the current message
                role = "Human" if msg["role"] == "user" else "Assistant"
                conversation_context += f"{role}: {msg['content']}\n"
        
        # Create the analysis prompt
        query_prompt = f"""You are analyzing a dataset for a user. Here is the dataset and their current question.

{context_section}

Dataset:
{converted_content}

{conversation_context}

Current question: {user_input}

Please provide a helpful analysis answering their question. Keep the conversation natural and reference previous questions if relevant."""
        
        try:
            analysis_response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert data analyst having a conversation with a user about their dataset. Provide clear, helpful analysis and maintain conversational context."},
                    {"role": "user", "content": query_prompt}
                ],
                temperature=0.3
            )
            
            analysis_result = analysis_response.choices[0].message.content.strip()
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": analysis_result})
            
            # Rerun to display the new message
            st.rerun()
            
        except Exception as e:
            st.error(f"Error analyzing data: {e}")
            
elif user_input and not converted_content:
    st.warning("Please upload and convert a file first.")

# Clear chat button
if st.session_state.chat_history:
    if st.button("Clear Chat", help="Clear the conversation history"):
        st.session_state.chat_history = []
        st.rerun()
