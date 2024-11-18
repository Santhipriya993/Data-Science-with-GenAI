from openai import OpenAI
import streamlit as st

f = open(r"C:\Users\Santhi\Desktop\keys\openai_api_key.txt")
OPENAI_API_KEY = f.read()

client = OpenAI(api_key = OPENAI_API_KEY)

st.title("💬An AI Code Reviewer")

def python_compiler(prompt):
    response = client.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content" : """I want you to act as a python compiler.You should analyze the generated code
                            and identify the potential bugs,errors.write the bugs with the format:
                            'The bugs in the code are: 1. ... 2. ... etc'with all identified bugs without line breaks.
                            'Fixed Code:' provide the fixed code snippet within text area."""
     
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    return response.choices[0].message.content


prompt = st.text_area("Enter your python code here...")
btn_click = st.button("Generate")

if btn_click:
    if prompt : 
        review_result = python_compiler(prompt)
        bug_report, fixed_code = review_result.split("Fixed Code:")
        st.header("Code Review")

        st.subheader("Bug Report")
        st.write(bug_report.strip())

        st.subheader("Fixed Code")
        st.write(fixed_code.strip())

    else:
        st.warning("please enter your python code")