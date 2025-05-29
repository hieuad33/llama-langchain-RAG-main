import streamlit as st
import os
from requestchatbot import ChatbotClient
from dotenv import load_dotenv
from find_pro import get_product_forchat
import matplotlib.pyplot as plt
from connectAPI_ABSA import analyze_product_api,predict_sentiment_api,predict_multilingual_api
load_dotenv()
from geminiAi import GeminiAIConnector
# App title
st.set_page_config(page_title="💬 Friends Chatbot")



id2label={'CAMERA',  'PERFORMANCE','FEATURES', 'DESIGN','PRICE','SCREEN','BATTERY','GENERAL','STORAGE','SER&ACC'}
emotions_MT={"Tích Cực":"POSITIVE","Tiêu Cực":"NEGATIVE","Bình Thường":"NEUTRAL"}

emotions_TM={"POSITIVE":"Tích Cực","NEGATIVE":"Tiêu Cực","NEUTRAL":"Bình Thường"}
cate_mt={'CAMERA':'CAMERA',
            'PERFORMANCE':'HIỆU SUẤT',
            'FEATURES':'CHỨC NĂNG',
                'DESIGN':'THIẾT KẾ',
                'SCREEN':'MÀN HÌNH',
                'BATTERY':'PIN',
                'GENERAL':'CHUNG',
                'PRICE':'GIÁ',
                'STORAGE':'BỘ NHỚ',
                'SER&ACC':'DỊCH VỤ'}
# Initialize ChatbotClient if not already in session_state
if 'gemmachat' not in st.session_state:
    st.session_state.gemmachat = ChatbotClient()



# Initialize session state to track the current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'chat'  # Default to the chat page

def clear_chat_history():
    if 'gemmachat' in st.session_state:
        st.session_state.gemmachat.clean_chat()
    st.session_state.messages = [{"role": "assistant", "content": "Hôm nay tôi có thể hỗ trợ gì cho bạn?"}]
    st.session_state.storychat = ""

# --- Sidebar ---
with st.sidebar:
    st.title('💬 Sale Chatbot')
    st.write('Chatbot này cung cấp một số biến thể của mô hình Llama 33 LLM + RAG. Hãy thoải mái để tôi giúp bạn tư vấn mua máy tính')

    # Navigation Buttons - Placed at the top
    if st.button('Chatbot'):
        st.session_state.current_page = 'chat'
    if st.button('ABSA Analysis'):
        st.session_state.current_page = 'absa'
    if st.button('ABSA Model Test'):
        st.session_state.current_page = 'testmodel'

    # Conditional Sidebar Content
    if st.session_state.current_page == 'chat':
        replicate_api_token_env = os.environ.get("REPLICATE_API_TOKEN")
        gemini_api_token_env= os.environ.get("GEMINI_API_TOKEN")

        # Obtain Credentials
        gemini_api = None
        if gemini_api_token_env:
            st.success('API key already provided from environment!', icon='✅')
            gemini_api = gemini_api_token_env
        else:
            gemini_api = st.text_input('Enter gemini API token:', type='password', key='gemini_api_input')
            if not (gemini_api and replicate_api.startswith('r8_')):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')
        replicate_api = None
        if replicate_api_token_env:
            st.success('API key already provided from environment!', icon='✅')
            replicate_api = replicate_api_token_env
        else:
            replicate_api = st.text_input('Enter Replicate API token:', type='password', key='replicate_api_input')
            if not (replicate_api and replicate_api.startswith('r8_') and len(replicate_api) == 40):
                st.warning('Please enter your credentials!', icon='⚠️')
            else:
                st.success('Proceed to entering your prompt message!', icon='👉')

        # Only set environment variables if replicate_api is valid
        if replicate_api and replicate_api.startswith('r8_') and len(replicate_api) >1:
            if os.environ.get('REPLICATE_API_TOKEN') != replicate_api:
                os.environ['REPLICATE_API_TOKEN'] = replicate_api
        if gemini_api and len(gemini_api) >5:
            if os.environ.get('GEMINI_API_TOKEN') != gemini_api:
                os.environ['GEMINI_API_TOKEN'] = gemini_api
        # Use session state for selected_option
        if 'gemini' not in st.session_state:
            st.session_state.gemini = GeminiAIConnector(gemini_api)
            print("GeminiAIConnector initialized with API key.",gemini_api)
        selected_option = st.selectbox(
            'Choose a model:',
            ['Gemma 2B','LLaMA3 with RAG','gemini','gemini with rag'],
            key='model_select',
            on_change=clear_chat_history,
            index=0
        )
        selected_option = st.session_state.model_select

        st.button('Clear Chat History', on_click=clear_chat_history)

# --- End Sidebar ---

# --- Main Content ---

# Conditional display based on session state
if st.session_state.current_page == 'chat':
    # --- Chat Interface ---
    st.header("Trò chuyện ngay")

    # Product Recommendations
    st.write("Sản phẩm được hệ thống đề xuất bạn có thể tham khảo")

    if 'products' not in st.session_state:
        st.session_state.products = get_product_forchat("sản phẩm chất lượng bán chạy ")[0]
    if 'storychat' not in st.session_state:
        st.session_state.storychat = ""

    if 'products' in st.session_state:
        col1, col2, col3, col4 = st.columns(4)
        product_cols = [col1, col2, col3, col4]  # Store columns in a list

        for i, col in enumerate(product_cols):
            if i < len(st.session_state.products) and st.session_state.products[i]:  # Check if product exists
                with col:
                    if st.session_state.products[i]["url_img"] != '':
                        st.image(st.session_state.products[i]["url_img"], use_container_width=True)
                    else:
                        st.image("D:\chatbot\llama-langchain-RAG-main\llama-langchain-RAG-main\chatbot-la-gi-co-nen-su-dung-chat-bot-de-chot-don.jpg", use_container_width=True)
                    st.text(st.session_state.products[i]["name"])
                    st.write(f"Giá: {st.session_state.products[i]['price']}")
                    st.markdown(f"[Xem chi tiết]({st.session_state.products[i]['url']})")
            else:
                with col:
                    st.text("Không có sản phẩm")  # Or some placeholder message
    else:
        st.text("không có sản phẩm nào được đề xuất")

    # Chat Messages
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Hôm nay tôi có thể hỗ trợ gì cho bạn?"}]

    chat_history_container = st.container(height=400, border=True)

    with chat_history_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Response Generation
    def generate_response(prompt_input):
        string_dialogue = "Bạn là một trợ lý hữu ích. Bạn hãy trả lời câu hỏi như một trợ lý hỗ trợ tư . \n"
        output = ""
        current_model = st.session_state.model_select
        
       
        if current_model == 'gemini with rag':
            try:
                from geminiAi import GeminiAIConnector
                output = st.session_state.gemini.generate_content(contents=f"{string_dialogue} {prompt_input}",rag=True)
            except ImportError:
                st.error("Error: query_rag function not found.")
                output = "Error: Could not generate response (query_rag not available)."
            except Exception as e:
                st.error(f"Error calling query_rag: {e}")
                output = f"Error: Could not generate response ({e})."
        elif current_model == 'gemini':
            try:
                from geminiAi import GeminiAIConnector
                output = st.session_state.gemini.generate_content(contents=f"{string_dialogue} {prompt_input}")
            except ImportError:
                st.error("Error: query_rag function not found.")
                output = "Error: Could not generate response (query_rag not available)."
            except Exception as e:
                st.error(f"Error calling query_rag: {e}")
                output = f"Error: Could not generate response ({e})."
        elif current_model == 'LLaMA3 with RAG':
            try:
                from query_data import query_rag
                output = query_rag(f"{string_dialogue} {prompt_input} Trợ lý trả lời ngắn gọn đầy đủ: ")
            except ImportError:
                st.error("Error: query_rag function not found.")
                output = "Error: Could not generate response (query_rag not available)."
            except Exception as e:
                st.error(f"Error calling query_rag: {e}")
                output = f"Error: Could not generate response ({e})."
        elif current_model == 'Gemma 2B':
            try:
                if 'gemmachat' in st.session_state:
                    output = st.session_state.gemmachat.chat(prompt_input)
                else:
                    st.error("Error: ChatbotClient not initialized.")
                    output = "Error: Could not generate response (Chat client not ready)."
            except AttributeError:
                st.error("Error: ChatbotClient missing chat method.")
                output = "Error: Could not generate response (Chat client not ready)."
            except Exception as e:
                st.error(f"Error calling gemmachat.chat: {e}")
                output = f"Error: Could not generate response ({e})."
        else:
            output = f"Error: Model '{current_model}' not supported."

        return output

    # User Input
    api_is_valid = ('replicate_api' in locals() and replicate_api and replicate_api.startswith('r8_') and len(replicate_api) == 40)

    if prompt := st.chat_input("Nhập tin nhắn của bạn:", disabled=not api_is_valid):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # Generate and display the response
    if st.session_state.messages[-1]["role"] == "user" and api_is_valid:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                prompt_to_send = st.session_state.messages[-1]["content"]
                response = generate_response(prompt_to_send)
                placeholder = st.empty()
                full_response = ''
                try:
                    for item in response:
                        full_response += item
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                except TypeError:
                    full_response = response
                    placeholder.markdown(full_response)
                except Exception as e:
                    st.error(f"Error during response streaming/display: {e}")
                    full_response = "Error generating response."
                    placeholder.markdown(full_response)
     
                #st.session_state.storychat = st.session_state.gemmachat.chat("Viết ngắn gọn : "+ prompt_to_send + " " + full_response + " " + st.session_state.storychat)
                st.session_state.products = get_product_forchat( prompt_to_send + " " + full_response )[0]
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
        st.rerun()

elif st.session_state.current_page == 'absa':
        # Khởi tạo session_state
    if 'product_info' not in st.session_state:
        st.session_state.product_info = None
    if 'comments' not in st.session_state:
        st.session_state.comments = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'product_url' not in st.session_state:
        st.session_state.product_url = ''
    if 'text_pre' not in st.session_state:
        st.session_state.text_pre = ''
    if 'blurred' not in st.session_state:
        st.session_state.blurred = False  # Track if comments should be blurred
   # --- ABSA Analysis Page ---
    st.header("Thông tin về đánh giá sản phẩm")
    st.write("Phân tích sản phẩm đang được khách hàng đánh giá như thế nào")

    # Nhập URL và số lượng bình luận
    product_url=st.text_input("Nhập URL sản phẩm:", st.session_state.product_url)

    
    if st.button("Phân tích bình luận"):
        product_inf=analyze_product_api(product_url)
        #print(product_inf['product_info'])
        if  'error' in product_inf:
            st.subheader("Không tìm thấy")
            print()
            st.write(product_inf['error'])
        else:
            st.session_state.product_url = product_url  # Lưu URL vào session_state
            st.session_state.product_info = product_inf['product_info'][0]#get_video_info(video_url)
            st.session_state.comments = product_inf['comments']#get_video_comments(video_url, max_comments=num_comments)
            st.session_state.analysis_results =product_inf['analysis_summary'] #analyze_comments(st.session_state.comments)
            st.session_state.blurred = False  # Reset blur state when re-analyzing
            st.session_state.predict = product_inf['product_info']
            

            # Tạo hai cột
            col1, col2 = st.columns([1, 1])  # Phần trái có tỉ lệ 1, phần phải có tỉ lệ 2
            with col1:
                st.subheader("Thông tin sản phẩm:")
                st.write(f"Sản phẩm: {st.session_state.product_info['title']}")
                st.write(f"Giá bán: {st.session_state.product_info['price']}")
                #iff  <--  'iff' này có vẻ là lỗi đánh máy, bỏ đi
                st.image(
                    st.session_state.product_info['img_url'],
                    width=600)  # Manually Adjust the width of the image as per requirement

                #st.write(f"Số lượt xem: ----")  # {st.session_state.product_info['view_count']}")

            with col2:
                st.subheader("Thông tin đánh giá:")

                # for   , percentage in st.session_state.predict:
                #     st.write(f"{cate_mt[category]}: {percentage}/10")
                
                fig, ax = plt.subplots(figsize=(4, 7))
                categories=[item[0] for item in st.session_state.analysis_results]
                values=[item[1] for item in st.session_state.analysis_results]
                
                # Vẽ cột 100% (nền)
                for i in range(len(categories)):
                    ax.barh(i, 10, left=0, height=0.8, color='white', edgecolor='black', linewidth=2)

                # Vẽ các cột dữ liệu (đè lên cột 100%)
                ax.barh(range(len(categories)), values, height=0.8, color='orange', edgecolor='black', linewidth=1)

                # Thiết lập nhãn trục y (hiển thị tên cột)d
                ax.set_yticks(range(len(categories)))
                ax.set_yticklabels(categories)

                # Thiết lập nhãn trục x
                for i, percentage in enumerate(values):
                    ax.text(10, i, f'{percentage}', va='center', ha='left', fontsize=10)

                # Thiết lập giới hạn trục x
                ax.set_xlim(0, 10)

                # Ẩn các cạnh của khung hình
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)

                # Tắt các vạch chia
                ax.tick_params(left=False, bottom=False)

                # Xoay nhãn trục x nếu cần thiết
                ax.set_xticklabels(ax.get_xticks(), rotation=45)

                # Hiển thị biểu đồ trong Streamlit
                st.pyplot(fig)

        #st.video(st.session_state.video_url)
    # #hiển thị bình luận
    if st.session_state.comments:
        
        st.subheader("Phân tích cảm xúc bình luận:")
        st.write(f"Số bình luận: {st.session_state.product_info['comment_count']}")

        # st.subheader("Phân tích cảm xúc bình luận:")
        #unique_emotions = ["Tất cả"] + list({result['emotion'] for result in st.session_state.analysis_results})
        unique_emotions = ["Tất cả", "Tích Cực", "Tiêu Cực", "Bình Thường"]  # list({result for result in st.session_state.analysis_results})
        selected_emotion = st.selectbox("Chọn cảm xúc để lọc bình luận:", unique_emotions)

        unique_cate = ['TẤT CẢ', 'CAMERA', 'HIỆU SUẤT', 'CHỨC NĂNG', 'THIẾT KẾ', 'GIÁ', 'MÀN HÌNH', 'PIN', 'CHUNG', 'BỘ NHỚ', 'DỊCH VỤ']  # list({result for result in st.session_state.analysis_results})
        selected_cate = st.selectbox("Chọn cảm danh mục bình luận:", unique_cate)

        if st.button("Chi Tiết"):
            st.session_state.blurred = not st.session_state.blurred

        st.session_state.countbl = 0

        for result in st.session_state.comments:

            if selected_cate != 'TẤT CẢ':
                kt = True
                for vtc in id2label:
                    if cate_mt[vtc] == selected_cate and result["predicts"][vtc] != "O":
                        kt = False
                if kt:
                    continue
            if selected_emotion != "Tất cả":
                kt = True
                for vtc in id2label:
                    if emotions_MT[selected_emotion] == result["predicts"][vtc]:
                        kt = False
                if kt:
                    continue
            st.session_state.countbl += 1
            if (st.session_state.blurred):
                st.write(f" {result['review']}")
                with st.expander("Xem chi tiết"):

                    for vt in id2label:
                        if result["predicts"][vt] != "O":
                            if (result["predicts"][vt] == "POSITIVE"):
                                st.write(f"- **{cate_mt[vt]}**: :blue[ {emotions_TM[result["predicts"][vt]]}]")
                            if (result["predicts"][vt] == "NEGATIVE"):
                                st.write(f"- **{cate_mt[vt]}**: :red[{emotions_TM[result["predicts"][vt]]}] ")
                            if (result["predicts"][vt] == "NEUTRAL"):
                                st.write(f"- **{cate_mt[vt]}**: {emotions_TM[result["predicts"][vt]]}")

        st.write(f"Tổng số bình luận tìm thấy: {st.session_state.countbl}")

if st.session_state.current_page=="testmodel":
    if 'text_pre' not in st.session_state:
        st.session_state.text_pre = ''
    st.title("Phân tích đánh giá của bạn")
    text = st.text_area("Nhập văn bản đánh giá:", st.session_state.text_pre)
    if st.button("Phân tích bình luận"):
        ketquadudoan=predict_sentiment_api(text)
        for vl,s in ketquadudoan.items():
                st.write(f"- **{vl}**: {s}")
        st.write("----------")