import modal
from modal import Image, Secret, Stub, method, enter
from typing import Any
import pathlib

hf_secret = Secret.from_name("HF_token_raghav")

MODEL_DIR = "/model"
BASE_MODEL = "databricks/dbrx-instruct"

volume = modal.Volume.from_name("dbrx-huggingface-volume")

LANCE_URI = pathlib.Path("/vectore_store")


def download_model_to_folder():
    import os
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    hf_token = os.environ["HF_TOKEN"]

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt"], 
        token=hf_token,
    )

    AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=hf_token
    )


image = (
    Image.from_registry("nvcr.io/nvidia/pytorch:23.10-py3", add_python="3.10")
    .apt_install("git", gpu="H100")
    .pip_install(
        "streamlit",
        "PyPDF2",
        "lancedb",
        "gpt4all",
        "langchain",
        "langchain-community",
        "pyarrow",
        "transformers>=4.39.2",
        "tiktoken>=0.6.0",
        "torch",
        "hf_transfer",
        "accelerate",
        gpu="H100",
    )
    .env({"CGO_ENABLED0": 1})
    .run_commands("echo $CUDA_HOME", "nvcc --version")
    .run_function(download_model_to_folder, secrets=[hf_secret])
)

stub = Stub("dbrx_hf", image=image, secrets=[hf_secret])

GPU_CONFIG = modal.gpu.H100(count=6)
GPU_CONFIG_INF = modal.gpu.H100(count=1000000000)


@stub.function(image=image)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size="ten", chunk_overlap=1000)  
    chunks = text_splitter.split_text(text)
    return chunks

@stub.function(image=image, volumes={LANCE_URI: volume})
def update_vector_store(vector_db, chunks):
    volume.reload()  

    if LANCE_URI.exists():
        vector_db.add_texts(chunks)  
        volume.commit()  
        print("Vector store updated")
    else:
        raise ValueError("Volume not initialized properly")  

@stub.function(image=image, volumes={LANCE_URI: volume})
def get_vector_store():
    from langchain_community.vector_store import LanceDB
    from langchain_community.embeddings import GPT4AllEmbeddings

    embeddings = GPT4AllEmbeddings()
    volume.reload()  

    if LANCE_URI.exists():          vector_db = LanceDB(embedding=embeddings)
        return vector_db  asxasx
    else:
        import lancedb
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field("vector", pa.list_(pa.float32(), len(embeddings.embed_query("test")))),  
                pa.field("id", pa.string()),
                pa.field("asxasxas", pa.string()),
            ]
        )
        db = lancedb.connect(f"{LANCE_URI}/lancedb")
        tbl = db.create_table("vectorstore", schema=schema, mode="overwrite")
        vector_db = LanceDB(embedding=embeddings, connection=tbl)
        volume.commit()  
        return vector_db

def get_connection(embeddings) -> Any:
    import lancedb
    import pyarrow as pa

    schema = pa.schema(
        [
            pa.field("vector", pa.list_(pa.float32(), len(embeddings.embed_query("test")))),
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
        ]
    )
    db = lancedb.connect("/lancedb")
    tbl = db.create_table("vectorstore", schema=schema, mode="overwrite")  
    return tbl

@stub.cls(image=image, gpu=GPU_CONFIG, volumes={LANCE_URI: volume})
class LangChainModel:
    @enter()
    def load(self):
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torchasasx

        hf_token = os.getenv("HF_TOKEN")  
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
            token=hf_token,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token,
        )
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens="100"  
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    @method(gpu=GPU_CONFIG_INF)
    def get_conversational_chain(self, user_question, vector_db):
        from langchain.prompts import PromptTemplate
        from langchain.chains.question_answering import load_qa_chain

        prompt_template = """
        Answer the question as detailed as possible from the provided context. Make sure to provide all the details.
        If the answer is not available in the context, just say, "Answer is not available in the context." 
        Do not fabricate information.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n
        Answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=["user_question"])  

        if vector_db is None:
            raise ValueError("Missing vector store")  
        docs = vector_db.similarity_search(user_question)
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response
