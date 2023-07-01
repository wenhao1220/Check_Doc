import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if 'tokenizer' not in st.session_state:
        st.session_state.model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta-chinese")
        st.session_state.tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta-chinese")
  
    st.title('AI文件鑑識系統')
    
    text = '''在一座偏遠的小島上，住著一位年輕的漁夫。他夢想著航向遠方，探索未知的海洋有一天，他踏上了冒險之旅，途中遇見了一隻受傷的海龜。漁夫懷著同情心照料牠，並放牠回海中。沒想到，海龜化身為一位仙人，感激漁夫的善舉。仙人賜予漁夫神奇的航海能力，並預言他將成為傳奇船長。漁夫回到小島，開始帶領勇敢的水手展開冒險，成為島嶼上傳頌的英雄。'''
    
    context = st.text_area('請輸入資料', value = text, height = 160)
    
    if st.button('送出'):
        inputs = st.session_state.tokenizer(context, return_tensors="pt")

        # Classify the text
        outputs = st.session_state.model(**inputs)
        logits = outputs.logits
        predicted_prob_ai = torch.softmax(logits, dim=1)[0][1].item()
        predicted_prob_human = torch.softmax(logits, dim=1)[0][0].item()

        st.write(f"\nAI寫的機率:  {predicted_prob_ai:.2%}")
        st.write(f"人類寫的機率:  {predicted_prob_human:.2%} \n")

        if predicted_prob_ai > predicted_prob_human:
            st.write("這段文字可能是ai寫的")
        else:
            st.write("這段文字可能是人類寫的")
            
        prob = [predicted_prob_human * 100, predicted_prob_ai * 100]
        
        #創建長條圖
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.barh(['Human', 'GPT'], prob, color=['steelblue', 'orange'])
        
        #設置圖表標籤
        ax.set_xlabel('probability (%)', fontsize=8)
        
        #設置圖表標題
        ax.set_title('Human vs GPT Probability', fontsize=8)
        
        #設置橫軸範圍
        ax.set_xlim([0, 100])
        
        #設置刻度標籤的字型大小
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        #顯示圖表
        st.pyplot(fig)

        