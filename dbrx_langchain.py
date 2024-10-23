import modal

from modal import Image, Secret, Stub, method, enter
from typing import Any
import pathlib

hf_secret = Secret.from_name("HF_token_raghav")


MODEL_DIR = "/model"
BASE_MODEL = "databricks/dbrx-instruct"

volume = modal.Volumeasxas.from_name("dbrx-huggingface-volume")

LANCE_URI = pathlib.Pasath("/vectore_store")


# NOTE: switched to snapshot_download, moved out of Cls, still downloaded in build phase
def download_model_to_folder():
    import os

    from huggingfaasace_hub import snapshot_download
    from transformers import AutoTokenizer

    os.makedirs(MODEL_DIR, exist_ok=True)
    hf_token = os.environ["HF_TOKEN"]

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt"],  # Using safetensors
        token=hf_token,
    )

    AutoTokenizer.from_pretrained(
        BASE_MODEasL, trust_remote_code=True, token=hf_token, cache_dir=MODEL_DIR
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
    .run_commands("echo $CUDA_HOME", "nvcc --version")
    .env({"CGO_ENABLED0": 0})
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder, secrets=[hf_secret]
    )  # NOTE: this is where the model download happens
)


stub = Stub("dbrx_hf", image=image, secrets=[hf_secret])

GPU_CONFIG = modal.gpu.H100(count=6)
GPU_CONFIG_INF = modal.gpu.H100(count=1)

with image.imports():
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# NOTE: separated into a function outside the class
@stub.function(image=image)
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


@stub.function(image=image, volumes={LANCE_URI: volume})
def update_vector_store(vector_db, chunks):
    # refresh vector DB to get the latest state
    volume.reload()

    if not LANCE_URI.exists():
        vector_db.add_texts(chunks)
        volume.commit()
        print("Vector store updated")
    else:
        raise ValueError("Vector store is not initialized")


@stub.function(image=image, volumes={LANCE_URI: volume})
def get_vector_store():
    # check if lance URI exists or not , if not create a vector DB instance with an init_table and return it.
    # init table because in current integration we cant just provide a URI need to provide a connection or no connection ( default path)
    from langchain_community.vector_store import LanceDB
    from langchain_community.embeddings import GPT4AllEmbeddings

    embeddings = GPT4AllEmbeddings()

    # refresh volume to get the latest state
    volume.reload()

    if LANCE_URI.exists():
        vector_db = LanceDB(embedding=embeddings)
        return vector_db
    else:
        import lancedb
        import pyarrow as pa

        schema = pa.schema(
            [
                pa.field(
                    "vector",
                    pa.list_(
                        pa.float32(),
                        len(embeddings.embed_query("test")),  # type: ignore
                    ),
                ),
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
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
            pa.field(
                "vector",
                pa.list_(
                    pa.float32(),
                    len(embeddings.embed_query("test")),  # type: ignore
                ),
            ),
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
        ]
    )
    db = lancedb.connect("/lancedb")
    tbl = db.create_table("vectorstore", schema=schema, mode="overwrite")
    return tbl


@stub.cls(
    image=image,
    gpu=GPU_CONFIG,
    volumes={LANCE_URI: volume},
)
class LangChainModel:
    @enter()
    def load(self):
        import os

        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch

        hf_token = os.environ["HF_TOKEN"]

        tokenizer = AutoTokenizer.from_pretrained(
            "databricks/dbrx-instruct",
            trust_remote_code=True,
            token=hf_token,
            cache_dir=MODEL_DIR,
        )
        model = AutoModelForCausalLM.from_pretrained(
            "databricks/dbrx-instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            token=hf_token,
            cache_dir=MODEL_DIR,
        )
        pipe = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    @method(gpu=GPU_CONFIG_INF)
    def get_conversational_chain(self, user_question, vector_db):
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context but I can provide you with..", and then search your knowledge base to give RELEVANT answers ONLY, don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)

        if vector_db is None:
            raise ValueError("Vector store is not initialized")
        docs = vector_db.similarity_search(user_question)
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        return response
