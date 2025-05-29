import requests
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from get_embedding_function import get_embedding_function
CHROMA_PATH = "./chroma"
kfi=2

class ChatbotClient:
    def __init__(self, base_url='http://localhost:8080'):
        self.base_url = base_url
        self.context = ""
        self.kfi=1
        self.assistant_response = ""
        embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    def clean_chat(self):
        self.context = ""
        self.assistant_response = ""
    def __rag_data(self, query_text):
        results = self.db.similarity_search_with_score(query_text, k=kfi)
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        print(sources)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return context_text
             
    def get_server_health(self):
        try:
            response = requests.get(f'{self.base_url}/health')
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            return f"Error connecting to the server: {e}"
        except ValueError:
            return "Error decoding JSON response."

    def __post_completion(self, context, user_input,rag_data):
        prompt = f"{context}\n dựa trên dữ liệu sau để trả lời: { rag_data} \n User: {user_input}\n Assistant:"
        print(prompt)
        data = {
            'prompt': prompt,
            'temperature': 0.8,
            'top_k': 35,
            'top_p': 0.95,
            'n_predict': 400,
            'stop': ["</s>", "Assistant:", "User:"]
        }
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post(f'{self.base_url}/completion', json=data, headers=headers)
            response.raise_for_status()
            return response.json()['content'].strip()
        except requests.exceptions.RequestException as e:
            return f"Error sending request: {e}"
        except (ValueError, KeyError):
            return "Error processing the server response."

    def __update_context(self, context, user_input, assistant_response):
        return f"{context}\nUser: {user_input}\nAssistant: {assistant_response}"
    def chat(self,user_input):
        rag_data=self.__rag_data(user_input)
        
        assistant_response = self.__post_completion(self.context, user_input,rag_data)
        if assistant_response and "Error" not in assistant_response:
            self.context = self.__update_context(self.context, user_input, assistant_response)
        return assistant_response
            
