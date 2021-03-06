from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st


st.set_page_config(
    page_title="CLimate Fact Checker",
    page_icon="🎈",
    layout = "centered"
)

st.title("Climate Fact Checker!")

with st.form(key="my_form", clear_on_submit=True):
  claim_type = st.selectbox(
                "Select claim type", ("Climate change", "Generic")
            )
  claim = st.text_input("Claim")
  evidence = st.text_input("Evidence")
  
  submitted = st.form_submit_button(label="Verify")

  if submitted:
      model = AutoModelForSequenceClassification.from_pretrained("amandakonet/climatebert-fact-checking")
      tokenizer = AutoTokenizer.from_pretrained("amandakonet/climatebert-fact-checking")
      features = tokenizer([str(claim)],[str(evidence)], padding='max_length', truncation=True, return_tensors="pt", max_length=512)

      model.eval()
      with torch.no_grad():
        scores = model(**features).logits
        label_mapping = ['SUPPORTED', 'REFUTED', 'NEI']
        labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
        st.write("CLAIM:", claim)
        st.write("EVIDENCE:", evidence)

        if labels[0]=='SUPPORTED':
          st.write('The evidence SUPPORTS the claim')
        if labels[0]=='REFUTED':
          st.write('The evidence REFUTES the claim')
        if labels[0]=='NEI':
          st.write('There isn\'t enough information to verify this claim')
