import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing import TypedDict

import os

# API KEY CHECK
if "GOOGLE_API_KEY" not in os.environ:
    st.error("GOOGLE_API_KEY not found. Please set it in Colab before running Streamlit.")
    st.stop()

st.set_page_config(page_title="Carbon Dashboard", layout="wide")

# LOAD DATA
df = pd.read_csv("individual_dataset.csv")

required_cols = [
    "Electricity_kWh",
    "Vehicle_km",
    "LPG_kg",
    "Water_liters"
]

for col in required_cols:
    if col not in df.columns:
        df[col] = 0

df = df.fillna(0)

# CARBON CALCULATIONS
df["Electricity_CO2"] = df["Electricity_kWh"] * 0.85
df["Vehicle_CO2"] = df["Vehicle_km"] * 0.21
df["LPG_CO2"] = df["LPG_kg"] * 2.98
df["Water_CO2"] = df["Water_liters"] * 0.002

df["Total_CO2"] = (
    df["Electricity_CO2"]
    + df["Vehicle_CO2"]
    + df["LPG_CO2"]
    + df["Water_CO2"]
)

# DEFAULT USER
current_user_df = df.iloc[[0]]

# OPTIMIZATION FUNCTION
def optimize_emissions(data):
    suggestions = []

    avg_elec = data["Electricity_kWh"].mean()
    avg_vehicle = data["Vehicle_km"].mean()
    avg_lpg = data["LPG_kg"].mean()
    avg_water = data["Water_liters"].mean()

    if avg_elec > 300:
        suggestions.append("Reduce electricity usage by switching off unused devices")

    if avg_vehicle > 100:
        suggestions.append("Use public transport or carpool")

    if avg_lpg > 8:
        suggestions.append("Reduce LPG usage by optimizing cooking efficiency")

    if avg_water > 1000:
        suggestions.append("Reduce water wastage")

    if not suggestions:
        suggestions.append("Your carbon footprint is already optimized")

    return suggestions

# LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

# LANGGRAPH STATE
class GraphState(TypedDict):
    question: str
    analysis: str
    suggestions: str
    answer: str

# NODES
def analysis_node(state):
    analysis = f"User CO2 emission is {current_user_df['Total_CO2'].values[0]:.2f}"
    return {"analysis": analysis}

def optimization_node(state):
    tips = optimize_emissions(current_user_df)
    return {"suggestions": "\n".join(tips)}

def chatbot_node(state):
    question = state["question"]
    analysis = state["analysis"]
    suggestions = state["suggestions"]

    prompt = f"""
You are a sustainability assistant.

User Question:
{question}

User Carbon Analysis:
{analysis}

Personalized Optimization Suggestions:
{suggestions}

Explain clearly and give helpful advice to the user.
"""

    try:
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        return {"answer": f"LLM Error: {str(e)}"}

# GRAPH BUILD
builder = StateGraph(GraphState)

builder.add_node("analysis", analysis_node)
builder.add_node("optimization", optimization_node)
builder.add_node("chatbot", chatbot_node)

builder.set_entry_point("analysis")

builder.add_edge("analysis", "optimization")
builder.add_edge("optimization", "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# ECOSCORE
max_emission = df["Total_CO2"].max()

df["EcoScore"] = (1 - (df["Total_CO2"] / max_emission)) * 100
df["EcoScore"] = df["EcoScore"].round(2)

# ALERT
threshold = 350

df["Alert"] = df["Total_CO2"].apply(
    lambda x: "⚠ High Emission" if x > threshold else "✅ Normal"
)

# SIDEBAR
st.sidebar.title("🌍 Carbon Dashboard")

section = st.sidebar.selectbox(
    "Choose Section",
    [
        "Dataset",
        "User Comparison",
        "Individual Analysis",
        "Emission Sources",
        "EcoScore",
        "AI Chatbot"
    ]
)

# TITLES
if section == "Dataset":
    st.title("📊 Carbon Dataset Overview")

elif section == "User Comparison":
    st.title("📈 User Emission Comparison")

elif section == "Individual Analysis":
    st.title("👤 Individual Carbon Analysis")

elif section == "Emission Sources":
    st.title("🌱 Emission Source Breakdown")

elif section == "EcoScore":
    st.title("🌿 EcoScore & Alerts")

elif section == "AI Chatbot":
    st.title("🤖 AI Sustainability Assistant")

# SECTIONS
if section == "Dataset":
    st.dataframe(df, hide_index=True)  # ✅ FIXED

elif section == "User Comparison":
    fig = px.bar(df, x="Name", y="Total_CO2", color="Total_CO2")
    st.plotly_chart(fig)

elif section == "Individual Analysis":
    user = st.selectbox("Select User", df["Name"])
    user_df = df[df["Name"] == user]
    current_user_df = user_df
    st.dataframe(user_df, hide_index=True)  # ✅ FIXED

elif section == "Emission Sources":
    source = df[
        [
            "Electricity_CO2",
            "Vehicle_CO2",
            "LPG_CO2",
            "Water_CO2"
        ]
    ].sum().reset_index()

    source.columns = ["Source", "CO2"]

    fig = px.pie(source, values="CO2", names="Source")
    st.plotly_chart(fig)

elif section == "EcoScore":
    fig = px.bar(df, x="Name", y="EcoScore", color="EcoScore")
    st.plotly_chart(fig)

    st.subheader("High Emission Alerts")
    st.dataframe(df[df["Alert"] == "⚠ High Emission"], hide_index=True)

elif section == "AI Chatbot":
    user = st.selectbox("Select User for Optimization", df["Name"])
    current_user_df = df[df["Name"] == user]

    st.write("### Selected User Data")
    st.dataframe(current_user_df, hide_index=True)  # ✅ FIXED

    st.subheader("🔧 Optimization Suggestions")

    tips = optimize_emissions(current_user_df)

    for tip in tips:
        st.write("•", tip)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Ask about sustainability...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.chat_message("user"):
            st.write(user_input)

        result = graph.invoke({"question": user_input})

        answer = result["answer"]

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

        with st.chat_message("assistant"):
            st.write(answer)
