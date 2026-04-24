import os
from langchain_core.tools import tool
from googleapiclient.discovery import build
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import wikipedia

@tool
def youtube_search(query: str) -> str:
    """youtube_search: Use this when the user is asking for videos, tutorials, music, or visual explanations. Returns relevant video titles and links."""
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        return "YouTube API key not found."
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=3
        )
        response = request.execute()
        results = []
        for item in response.get('items', []):
            video_id = item['id']['videoId']
            title = item['snippet']['title']
            url = f"https://www.youtube.com/watch?v={video_id}"
            results.append(f"Title: {title}\nURL: {url}")
        return "\n\n".join(results) if results else "No videos found."
    except Exception as e:
        return f"Error using YouTube API: {str(e)}"

@tool
def web_search(query: str) -> str:
    """web_search: Use this to search the internet for facts, history, or general knowledge using Wikipedia."""
    try:
        # Search for page titles
        results = wikipedia.search(query, results=1)
        if not results:
            return "No web search results found."
            
        page = wikipedia.page(results[0], auto_suggest=False)
        return f"Title: {page.title}\nSummary: {page.summary[:1500]}\nURL: {page.url}"
    except Exception as e:
        return f"Error searching the web: {str(e)}"

@tool
def rag(query: str) -> str:
    """rag: Use this when the query refers to internal/company/project documents or stored files. For example: explaining details from the user's resume or internal knowledge base."""
    if not os.path.exists("faiss_index"):
        return "Knowledge base not initialized. Please run ingest.py first."
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(query, k=3)
        if not docs:
            return "No relevant information found in the documents."
        return "\n\n".join([f"Document excerpt:\n{d.page_content}" for d in docs])
    except Exception as e:
        return f"Error querying local Knowledge Base: {str(e)}"

# List of tools to pass to the agent
AGENT_TOOLS = [youtube_search, web_search, rag]
