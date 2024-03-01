import streamlit as st
from openai import OpenAI

def generate_lyrics(artist_name, client):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "I want to generate song lyrics for " + artist_name + ", but do not put " + artist_name + "in it and do it in the artist language."}
        ]
    )

    generated_lyrics = response.choices[0].message.content.strip()

    return generated_lyrics


def main():
    st.title("GhoatWritter")
    st.markdown("Enter an artist !")

    artist_name = st.text_input("You: ", "")

    client = OpenAI()

    if st.button("Send"):
        bot_response = generate_lyrics(artist_name, client)
        st.text("GhoatWritter: " + bot_response)

if __name__ == "__main__":
    main()
