import os, re, ast, threading, json
import networkx as nx
from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import pyarrow as pa
import pyarrow.ipc as ipc

client = OpenAI()
SHORT_TERM_WINDOW = 15

class MemoryAdapterBase:
    def save_graph(self, graph: nx.DiGraph): raise NotImplementedError
    def load_graph(self) -> nx.DiGraph: raise NotImplementedError
    def save_embeddings(self, summaries: list[str]): raise NotImplementedError
    def load_embeddings(self): raise NotImplementedError
    def add_embeddings(self, new_summaries: list[str]): raise NotImplementedError
    def search(self, query: str, top_k: int = 5) -> list[str]: raise NotImplementedError

class LocalFileAdapter(MemoryAdapterBase):
    def __init__(self, profile_name="default", embeddings=None):
        self.profile_name = profile_name
        self.profile_dir = os.path.join("data", profile_name)
        os.makedirs(self.profile_dir, exist_ok=True)
        self.kg_path = os.path.join(self.profile_dir, f"memory_{profile_name}.arrow")
        self.faiss_path = os.path.join(self.profile_dir, f"faiss_{profile_name}")
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.vector_db = None

    def save_graph(self, graph: nx.DiGraph):
        graph_dict = nx.node_link_data(graph)
        table = pa.table({"graph": [json.dumps(graph_dict)]})
        with pa.OSFile(self.kg_path, "wb") as sink:
            with ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

    def load_graph(self) -> nx.DiGraph:
        if os.path.exists(self.kg_path):
            try:
                with pa.memory_map(self.kg_path, "r") as source:
                    reader = ipc.open_file(source)
                    table = reader.read_all()
                    graph_json = table["graph"][0].as_py()
                    graph_dict = json.loads(graph_json)
                    return nx.node_link_graph(graph_dict)
            except Exception as e:
                print(f"Arrow load failed: {e}")
        return nx.DiGraph()

    def add_embeddings(self, new_summaries: list[str]):
        """Add summaries to FAISS in a separate thread."""
        def _update():
            try:
                if os.path.exists(self.faiss_path):
                    db = FAISS.load_local(
                        self.faiss_path, self.embeddings, allow_dangerous_deserialization=True
                    )
                    db.add_texts(new_summaries)
                else:
                    db = FAISS.from_texts(new_summaries, self.embeddings)
                db.save_local(self.faiss_path)
                self.vector_db = db
            except Exception as e:
                print(f"FAISS update failed: {e}")

        threading.Thread(target=_update, daemon=True).start()

    def load_embeddings(self):
        if os.path.exists(self.faiss_path):
            try:
                self.vector_db = FAISS.load_local(
                    self.faiss_path, self.embeddings, allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"FAISS load failed: {e}")

    def search(self, query: str, top_k: int = 5) -> list[str]:
        if self.vector_db:
            results = self.vector_db.similarity_search(query, k=top_k)
            return [r.page_content for r in results]
        return []

class MemoryKG:
    def __init__(self, adapter: MemoryAdapterBase, profile_name="default"):
        self.client = OpenAI()
        self.profile_name = profile_name
        self.adapter = adapter
        self.G = self.adapter.load_graph()
        self.adapter.load_embeddings()
        self.node_counter = len(self.G.nodes)

    def _extract_triplets_chunk(self, messages_chunk):
        """Extract (subject, predicate, object) triplets from message text."""
        text = "\n".join(
            m["content"] for m in messages_chunk if m["role"] in ["user", "assistant"]
        )
        if not text.strip():
            return []
        prompt = (
            "Extract concise factual (subject, predicate, object) triplets from the text below. "
            "Return ONLY a valid Python list of tuples.\n\nText:\n"
            f"{text}"
        )
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            raw = resp.choices[0].message.content
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                triplets = ast.literal_eval(match.group())
                normalized = []
                for t in triplets:
                    if isinstance(t, (tuple, list)) and len(t) == 3:
                        normalized.append(tuple(t))
                    elif isinstance(t, dict) and {"subject", "predicate", "object"}.issubset(t.keys()):
                        normalized.append((t["subject"], t["predicate"], t["object"]))
                return normalized
        except Exception as e:
            print(f"Triplet extraction failed: {e}")
        return []

    def _get_or_create_node(self, label):
        for n, data in self.G.nodes(data=True):
            if data.get("label") == label:
                return n
        node_id = f"entity_{self.node_counter}"
        self.node_counter += 1
        self.G.add_node(node_id, type="Entity", label=label)
        return node_id

    def add_chunk_to_graph(self, new_messages, photo_name=None):
        """Add messages to KG. Optionally tag by photo."""
        clean_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in new_messages if m["role"] in ["user", "assistant"]
        ]
        if not clean_messages:
            return

        triplets = self._extract_triplets_chunk(clean_messages)
        new_summaries = []

        for s, p, o in triplets:
            s_id = self._get_or_create_node(s)
            o_id = self._get_or_create_node(o)
            relation = p
            if photo_name:
                relation = f"{p} [photo: {photo_name}]"
            self.G.add_edge(s_id, o_id, relation=relation)
            new_summaries.extend([s, o])

        self.adapter.save_graph(self.G)
        if new_summaries:
            self.adapter.add_embeddings(new_summaries)

    def retrieve_relevant_context(self, query, top_k=5):
        """
        Return readable context: both semantic matches and structured triplets.
        This feeds GPT context so it can 'remember' facts from graph.
        """
        text_hits = self.adapter.search(query, top_k)
        edge_context = []
        for u, v, d in list(self.G.edges(data=True))[-20:]:
            edge_context.append(
                f"{self.G.nodes[u].get('label')} - {d.get('relation', '')} -> {self.G.nodes[v].get('label')}"
            )

        context_str = ""
        if text_hits:
            context_str += "Semantic memory:\n" + "\n".join(text_hits)
        if edge_context:
            context_str += "\n\nRecent graph relations:\n" + "\n".join(edge_context[-10:])
        return context_str.strip()

    def display_streamlit(self, height=600, width=800):
        """Display the graph interactively in Streamlit."""
        try:
            from streamlit.components.v1 import html
            from pyvis.network import Network

            net = Network(height=f"{height}px", width=f"{width}px", notebook=False, directed=True)
            nodes_to_display = [(n, self.G.nodes[n].get("label", n)) for n in self.G.nodes][-100:]
            node_ids = {n for n, _ in nodes_to_display}

            for n, label in nodes_to_display:
                net.add_node(n, label=label, title=label, color="lightblue")

            for u, v, d in self.G.edges(data=True):
                if u in node_ids and v in node_ids:
                    net.add_edge(u, v, title=d.get("relation", ""))

            net.repulsion(node_distance=200, central_gravity=0.3)
            temp_file = os.path.join("data", self.profile_name, "temp_memory_kg.html")
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            net.save_graph(temp_file)
            with open(temp_file, "r", encoding="utf-8") as f:
                html(f.read(), height=height)
        except Exception as e:
            print(f"Failed to render graph: {e}")