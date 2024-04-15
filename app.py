import streamlit as st

def process_text(text):
    """Converts text to uppercase and counts the characters."""
    uppercase_text = text.upper()
    char_count = len(uppercase_text)
    return uppercase_text, char_count

def main():
    st.title("Video Finder")
    data = st.text_input("What would you like to watch?")
    if st.button("Process"):
        processed_text, char_count = process_text(data)
        st.write("Processed Data:", processed_text)
        st.write("Character Count:", char_count)

if __name__ == '__main__':
    main()