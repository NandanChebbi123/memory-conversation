import streamlit as st
from openai import OpenAI
import base64
import os
import json
from pkg.memory_kg import MemoryKG, LocalFileAdapter
from langchain.schema import HumanMessage, AIMessage

client = OpenAI()
SHORT_TERM_WINDOW = 15

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def flatten_content(content):
    if isinstance(content, list):
        return " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content)

def limit_text_length(text: str, max_chars: int = 500) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text

def retrieve_relevant_memory(query: str, memory_kg: MemoryKG, k=5) -> str:
    """Retrieve long-term memory context for GPT"""
    return memory_kg.retrieve_relevant_context(query, top_k=k) or ""

def format_conversation(conv: list) -> list[dict]:
    formatted = []
    for m in conv:
        if isinstance(m["content"], list):
            text = " ".join(
                c.get("text", "") if isinstance(c, dict) else str(c)
                for c in m["content"]
            )
        else:
            text = str(m["content"])
        formatted.append({"role": m["role"], "content": text.strip()})
    return formatted

def send_to_openai(conversation: list, max_chars: int = 500) -> str:
    try:
        limit_instruction = f"\n\n[IMPORTANT RULE]: Reply must be under {max_chars} chars, end naturally."
        has_system = any(msg.get("role") == "system" for msg in conversation)
        if has_system:
            for msg in conversation:
                if msg.get("role") == "system":
                    msg["content"] += limit_instruction
        else:
            conversation.insert(0, {
                "role": "system",
                "content": "You are a warm, natural photo companion." + limit_instruction
            })

        response = client.responses.create(
            model="gpt-4.1",
            input=conversation,
            max_output_tokens=300
        )
        reply = response.output[0].content[0].text.strip()
        if len(reply) > max_chars + 10:
            reply = reply[:max_chars] + "..."
        return reply
    except Exception as e:
        return f"[Error communicating with GPT-4.1: {e}]"

def initialize_profile_state(profile: str, profile_dir: str):
    keys = {
        "selected_image": f"selected_image_{profile}",
        "image_initialized": f"image_initialized_{profile}",
        "messages": f"messages_{profile}",
        "conversation": f"conversation_{profile}",
        "memory_kg": f"memory_kg_{profile}"
    }

    st.session_state.setdefault(keys["selected_image"], None)
    st.session_state.setdefault(keys["image_initialized"], None)
    st.session_state.setdefault(keys["messages"], [])

    if keys["conversation"] not in st.session_state:
        conv_file = os.path.join(profile_dir, "conversation.json")
        if os.path.exists(conv_file):
            with open(conv_file, "r", encoding="utf-8") as f:
                st.session_state[keys["conversation"]] = json.load(f)
        else:
            st.session_state[keys["conversation"]] = []

    if keys["memory_kg"] not in st.session_state:
        adapter = LocalFileAdapter(profile_name=profile)
        st.session_state[keys["memory_kg"]] = MemoryKG(adapter=adapter, profile_name=profile)

    return keys

def run(image_path: str = None):
    st.title("MindLink")
    st.write("Upload images and chat with the assistant!")

    if "active_profile" not in st.session_state:
        st.warning("Please select a profile first!")
        return

    profile = st.session_state.get("active_profile", "default")
    profile_dir = os.path.join("data", "profiles", profile)
    os.makedirs(profile_dir, exist_ok=True)

    keys = initialize_profile_state(profile, profile_dir)
    messages_key = keys["messages"]
    conversation_key = keys["conversation"]
    image_key = keys["image_initialized"]
    memory_key = keys["memory_kg"]
    selected_image_key = keys["selected_image"]

    memory_kg: MemoryKG = st.session_state[memory_key]

    if image_path:
        st.session_state[selected_image_key] = image_path

    selected_image = st.session_state.get(selected_image_key)
    if selected_image and os.path.exists(selected_image):
        last_image = st.session_state.get(image_key)
        if last_image != selected_image:
            st.session_state[image_key] = selected_image
            st.session_state["photo_intro_sent"] = False
            st.info(f"📸 New photo selected: {os.path.basename(selected_image)}")

        st.image(selected_image, caption="Selected Photo", use_container_width=True)
        base64_image = encode_image(selected_image)

        short_term = st.session_state[conversation_key][-SHORT_TERM_WINDOW:]
        if short_term:
            formatted_msgs = [
                {"role": m.get("role", "user"), "content": flatten_content(m.get("content", ""))}
                for m in short_term if isinstance(m, dict)
            ]
            if formatted_msgs:
                memory_kg.add_chunk_to_graph(formatted_msgs)

        memory_summary = retrieve_relevant_memory(
            "Describe context for this photo", memory_kg
        )

        if not st.session_state.get("photo_intro_sent", False):
            system_prompt = "You are a warm, empathetic photo companion. Keep replies concise."
            if short_term:
                system_prompt += f"\nShort-term memory: {format_conversation(short_term)}"
            if memory_summary:
                system_prompt += f"\nLong-term memory: {memory_summary}"

            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"This is a new photo: {os.path.basename(selected_image)}. Let's talk about it."},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                    ],
                },
            ]

            reply = send_to_openai(prompt_messages)
            reply = limit_text_length(reply, 520)

            st.session_state[conversation_key].append({
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"This is a new photo: {os.path.basename(selected_image)}. Let's talk about it."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            })
            st.session_state[conversation_key].append({
                "role": "assistant",
                "content": [{"type": "output_text", "text": reply}]
            })
            st.session_state[messages_key].append(HumanMessage(content=f"Let's talk about {os.path.basename(selected_image)}"))
            st.session_state[messages_key].append(AIMessage(content=reply))
            st.session_state["photo_intro_sent"] = True

    user_input = st.chat_input("Your message")
    if user_input and user_input.strip():
        st.session_state[conversation_key].append({
            "role": "user",
            "content": [{"type": "input_text", "text": user_input}]
        })

        short_term = st.session_state[conversation_key][-SHORT_TERM_WINDOW:]
        memory_summary = retrieve_relevant_memory(user_input, memory_kg)

        system_prompt = "You are a warm, empathetic photo companion."
        if short_term:
            system_prompt += f"\nShort-term memory: {format_conversation(short_term)}"
        if memory_summary:
            system_prompt += f"\nLong-term memory: {memory_summary}"

        truncated_conversation = format_conversation(short_term)
        prompt_messages = [{"role": "system", "content": system_prompt}] + truncated_conversation

        reply = send_to_openai(prompt_messages)
        reply = limit_text_length(reply, 500)

        st.session_state[conversation_key].append({
            "role": "assistant",
            "content": [{"type": "output_text", "text": reply}]
        })
        st.session_state[messages_key].append(HumanMessage(content=user_input))
        st.session_state[messages_key].append(AIMessage(content=reply))

        try:
            formatted_msgs = [
                {"role": m.get("role", "user"), "content": flatten_content(m.get("content", ""))}
                for m in short_term if isinstance(m, dict)
            ]
            if formatted_msgs:
                memory_kg.add_chunk_to_graph(formatted_msgs)
        except Exception as e:
            st.error(f"Failed to update memory graph: {e}")

    st.subheader("Chat")
    for msg in st.session_state[messages_key]:
        sender = "You" if isinstance(msg, HumanMessage) else "Assistant"
        st.markdown(
            f"**{sender}:** {msg.content}" if sender == "You" else
            f"<span style='color:blue'><b>{sender}:</b> {msg.content}</span>",
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
    image_path = None
    if uploaded_file:
        profile = st.session_state.get("active_profile", "default")
        profile_dir = os.path.join("data", "profiles", profile)
        os.makedirs(profile_dir, exist_ok=True)
        image_path = os.path.join(profile_dir, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    run(image_path=image_path)