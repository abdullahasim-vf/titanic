import streamlit as st
import pandas as pd
import plotly.express as px

from titanic_rag_chatbot import load_data, prepare_full_context, ask_llm

st.set_page_config(page_title="Titanic Chat", page_icon="🛳️")
st.title("🛳️ Titanic Dataset Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I'm a Titanic expert. Ask me anything about the passengers!"})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

@st.cache_resource
def prepare_data():
    df = load_data()
    context = prepare_full_context(df)
    return df, context

df, full_context = prepare_data()


def extract_column(user_prompt, df_columns):
    """Extract mentioned column from prompt."""
    prompt_lower = user_prompt.lower()
    for col in df_columns:
        if col.lower() in prompt_lower:
            return col
    return None


def plot_requested_graph(user_prompt, df):
    """Plot graph if user requests one."""
    prompt_lower = user_prompt.lower()
    col = extract_column(prompt_lower, df.columns)

    if not col:
        return False

    if "scatter" in prompt_lower:
        fig = px.scatter(df, x=col, y='Fare', color='Survived', title=f"Scatter Plot: {col} vs Fare")
        st.plotly_chart(fig)
        return True

    if "histogram" in prompt_lower:
        fig = px.histogram(df, x=col, title=f"Histogram of {col}")
        st.plotly_chart(fig)
        return True

    if "pie" in prompt_lower:
        fig = px.pie(df, names=col, title=f"Pie Chart of {col}")
        st.plotly_chart(fig)
        return True

    if "bar chart" in prompt_lower or "bar graph" in prompt_lower:
        fig = px.bar(df, x=col, y='Fare', title=f"Bar Chart: {col} vs Fare")
        st.plotly_chart(fig)
        return True

    return False


if prompt := st.chat_input("Your question about Titanic passengers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner('Thinking...'):
            try:
                graph_drawn = plot_requested_graph(prompt, df)

                if not graph_drawn:
                    answer = ask_llm(prompt, full_context)
                    st.markdown(answer)
                    response_text = answer
                else:
                    response_text = "Here is your requested graph."

                with st.expander("View full dataset context used"):
                    st.write(full_context)

                st.session_state.messages.append({"role": "assistant", "content": response_text})

            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
