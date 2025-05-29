from google import generativeai as genai
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
CHROMA_PATH = "./chroma"

class GeminiAIConnector:
    """
    Một lớp để kết nối và tương tác với Gemini AI.
    """
    def __init__(self, api_key: str):
        """
        Khởi tạo GeminiAIConnector với khóa API.

        Args:
            api_key (str): Khóa API cho Gemini AI.
        """
        print("api_key !!", api_key)
        if not api_key:
            raise ValueError("API Key không được để trống.")
        genai.configure(api_key=api_key)
        self.client = genai
        embedding_function = get_embedding_function()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        
    def get_Rag_data(self,contents,kfi=2):
        results = self.db.similarity_search_with_score(contents, k=kfi)
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        print(sources)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return context_text
    def generate_content(self, contents: str, model_name="gemini-2.0-flash", rag=False):
        """
        Tạo nội dung bằng cách sử dụng một mô hình Gemini cụ thể.

        Args:
            contents (str): Nội dung đầu vào để tạo phản hồi.
            model_name (str): Tên của mô hình Gemini cần sử dụng (ví dụ: "gemini-pro", "gemini-2.0-flash").
            rag (bool): Xác định có sử dụng RAG (Retrieval-Augmented Generation) hay không.

        Returns:
            str: Văn bản phản hồi được tạo bởi mô hình.

        Raises:
            Exception: Nếu có lỗi trong quá trình tạo nội dung.
        """
        if rag:
            data_Rag = self.get_Rag_data(contents)
            contents = contents + " dựa trên nội dung dưới đây nếu phù hợp và trả lời câu hỏi 1 cách ngắn gọn: " + data_Rag
        else:
             contents = contents + " trả lời câu hỏi 1 cách ngắn gọn"
        try:
            model = self.client.GenerativeModel(model_name)
            response = model.generate_content(contents)
            return response.text
        except Exception as e:
            print(f"Đã xảy ra lỗi khi tạo nội dung: {e}")
            raise
