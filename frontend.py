import streamlit as st
import requests

# Title of the Streamlit app
st.title("Titanic Chatbot")

# Input field for the user's question
question = st.text_input("Ask a question about the Titanic dataset:")

# Button to submit the question
if st.button("Submit"):
    if question:
        # Send the question to the FastAPI backend
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"question": question}
            )
            # Check if the request was successful
            if response.status_code == 200:
                # Display the chatbot's answer
                answer = response.json()["answer"]
                st.write(f"**Answer:** {answer}")
            else:
                st.error("Failed to get a response from the chatbot.")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Please ensure the backend is running.")
    else:
        st.warning("Please enter a question.")