import streamlit as st
from utils import load_chroma_db, load_llm, create_qa_chain, create_prompt_template
from utils import generate_response_with_context

chroma_db = load_chroma_db()
llm = load_llm()
PROMPT = create_prompt_template()
qa = create_qa_chain(chroma_db, llm, PROMPT)


st.title("Acıbadem Üniversitesi Asistanı")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


query = st.text_input("Sorunuzu yazın:")

answer = None

if st.button("Yanıt Al"):
    if query:
        try:
            result = generate_response_with_context(query, qa, chroma_db)
            answer = result["response"]
            sources = result["source_documents"]

            st.session_state.chat_history.insert(0, {
                "question": query,
                "answer": answer,
                "sources": sources
            })

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

    else:
        st.warning("Lütfen bir soru yazın!")

if answer:
    st.subheader("Yanıt:")
    st.write(answer)
    st.markdown("**Kaynak PDF(ler):**")
    for doc in sources:
        st.markdown(f"- `{doc['source']}`")



st.subheader("Sohbet Geçmişi")
for chat in st.session_state.chat_history:
    st.write(f"**Soru:** {chat['question']}")
    st.write(f"**Yanıt:** {chat['answer']}")
    if "sources" in chat:
        st.markdown("**Kaynak PDF(ler):**")
        for doc in chat["sources"]:
            st.markdown(f"- {doc['source']}")
    st.write("---")
