from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

def read_pdf_and_create_chunks(pdf_path, chunk_size=512, chunk_overlap=50):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_text(text)


def add_pdf_to_chroma_db(pdf_path, chroma_db, chunk_size=512, chunk_overlap=50, max_batch_size=166):
    existing_data_count = chroma_db._collection.count()
    print(f"Mevcut ChromaDB Kayıt Sayısı: {existing_data_count}")

    chunks = read_pdf_and_create_chunks(pdf_path, chunk_size, chunk_overlap)
    print(f"{pdf_path} içinden {len(chunks)} adet parça üretildi!")

    if len(chunks) > 0:

        for i in range(0, len(chunks), max_batch_size):
            batch = chunks[i:i + max_batch_size]
            metadatas = [{"source": os.path.basename(pdf_path)} for _ in batch]
            chroma_db.add_texts(batch, metadatas=metadatas)
            print(f"{len(batch)} chunk başarıyla ChromaDB'ye eklendi.")

        print(f"Güncellenmiş ChromaDB veri sayısı: {chroma_db._collection.count()}")


def create_embeddings():
    return OllamaEmbeddings(model="bge-m3:latest")

CHROMA_DB_PATH = "/Users/selinaydin/PycharmProjects/phi4/chroma_index"

def load_chroma_db():
    embedding_model = OllamaEmbeddings(model="bge-m3:latest")

    chroma_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model
    )

    return chroma_db


def test_chroma_db():
    chroma_db = load_chroma_db()
    print("Veritabanındaki toplam vektör sayısı:", chroma_db._collection.count())

test_chroma_db()


def load_llm():
    return ChatOllama(
        model="phi4",
        temperature=0.4,
        max_tokens=524
    )

#######################################################################
def create_prompt_template():
    prompt_template = """
       İnsan: Aşağıdaki bağlamı kullanarak soruya detaylı bir şekilde ve anlamlı bir şekilde yanıt ver. Eğer cevabı bilmiyorsan, bunu belirt ve uydurma. Cevabında sadece Türkçe dilini kullan, sana verdiğim veriler dışında herhangi bir şey ekleme veya çıkarma yapma.Türkçe olmayan kelimeler, harfler kullanma. Lütfen verilen bilgileri tam anlamıyla yorumla ve sadece o bilgileri kullanarak yanıt ver. Cevaplarını lütfen sadece sana verilen bilgiler ile ver. Ekstra kendin bir şey katma çünkü resmi bir kurum için resmi bir dil ile doğru bir şekilde yanıtlaman gerekiyor. Sana verilen veri dışına çıkmadan yanıtlaman lazım. Okul kartım kayboldu ne yapabilirim gibi sorulara düz sana soru-cevaplarda ne verilmiş ise onu yaz ekstra bir şey katma. Soruyu anlamadıysan ya da bilmiyorsan bilmediğini ya da anlamadığını belirt ve cevaplama. Soru soran insanları yanlış yönlendiremeyiz, önemli sorular soruyorlar ve doğru cevap almaları lazım. Sana verilen verilerde zaten kesin cümleler ve içerikler var bunların dışına çıkman yanlış olur. Lütfen bunlara dikkat ederek cevapla soruları. Başka okulara göre cevap verme lütfen. Sadece Acıbadem Üniversitesi bazında doğru cevaplar vermen gerekiyor. Okulumuzun EYS, OBS ve BADEMNET platformları var. Buralardan bilgilere ulaşıyoruz. Sadece sana verilen veriler doğrultusunda hareket et. Sana verilen pdf dosyalarına göre cevaplaman lazım yoksa verdiğin bilgiler yanlış olacak ve yanlış yönlendireceğiz okuldaki öğrencileri. Sen arkadaş canlısı Acıbadem Üniversitesi Öğrenci İşleri Asistanısın.

    <bağlam>
    {context}
    </bağlam>

    Soru: {question}

    Yanıtın aşağıdaki gibi olmalı:
    - **Kapsayıcı Olmalı**: Soruyu tamamen anlamalı ve verilen bağlama dayanarak en doğru cevabı üretmelisin.
    - **Kesin Olmalı**: Verdiğim bağlamda yer alan herhangi bir bilgiyi değiştirmemeli veya eksik belirtmemelisin.
    - **Türkçe Olmalı**: Tüm yanıtların Türkçe dilinde olmalı. Yanıttaki dilin, bağlama uygun olmalı ve Türkçeye sadık kalmalı.
    - **Bağlama Sadık Olmalı**: Bağlamda verilen veriler dışına çıkmamalısın. Soruyu yanıtlarken yalnızca verilen bağlamı kullanmalı ve herhangi bir spekülasyon yapmamalısın.

    Cevabında aşağıdaki öğeleri kullanarak, anlamlı ve ayrıntılı bir yanıt oluştur:
    - Önceki bilgilerden çıkarımlar yaparak, daha önce verilmiş olan verilerle tutarlı ve mantıklı bir yanıt oluştur.
    - Eğer verilen bilgi eksikse veya yanlışsa, bunu nazikçe belirterek, kesinlikle doğru olmayan bir bilgi eklememelisin.
    -Lütfen sana verilmeyen herhangi bir bilgiyi sallama.
    - Öğrenci İşleri mail adresi : ogrenci.isleri@acibadem.edu.tr
    -Öğrenci İşleri Telefon Numarası: 0216 250 3580 / 7703
                                    0 216 500 4240 / 4241 / 4243 / 4244 / 4245 / 4246 / 4248
    -Sana verilen bilgiye yönelik cevaplar ver ve sallama,uydurma ya da kendine göre cevaplama.Okul ile ilgili sana ne verildiyse onu cevapla

    Asistan:
        """
    return PromptTemplate(template=prompt_template, input_variables=["context", "question"])
#################################################################################################

def create_qa_chain(chroma_db, llm, prompt_template):
    retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3,
                       },
        embedding_function=OllamaEmbeddings(model="bge-m3:latest")
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

def generate_response_with_context(query, qa, chroma_db):
    retriever = chroma_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3},
        embedding_function=create_embeddings()
    )

    relevant_docs = retriever.invoke(query)
    context = " ".join([doc.page_content for doc in relevant_docs])

    response = qa.invoke({"context": context, "query": query})

    return {
        "response": response.get("result", "Yanıt bulunamadı."),
        "context_used": context,
        "source_documents": [
    {
        "content": doc.page_content,
        "source": doc.metadata.get("source", "Bilinmiyor"),
    }
    for doc in relevant_docs
]

    }



if __name__ == "__main__":
    source_dir = "/Users/selinaydin/PycharmProjects/phi4/source_doc"
    chroma_db = load_chroma_db()


    pdf_files = [os.path.join(source_dir, file) for file in os.listdir(source_dir) if file.endswith(".pdf")]

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path} ...")
        add_pdf_to_chroma_db(pdf_path, chroma_db)

    print("Tüm PDF'ler başarıyla eklendi ve ChromaDB güncellendi!")
