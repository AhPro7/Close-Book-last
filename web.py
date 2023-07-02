import streamlit as st
import os
from model import TextGenerator
import pandas as pd

text_generator = TextGenerator()

st.set_page_config(
    page_title="Close Book",
    page_icon="👋",
)

st.sidebar.title('Available Books')
st.sidebar.caption('put your books in Books folder the books should be pdfs and in folder(as many as editions) with the same name')
books = os.listdir("Books")
book_name = st.sidebar.selectbox('Select a book', books)


st.sidebar.title('Put your urls \n spllited by new line')
urls_text = st.sidebar.text_area('Enter urls:', 'link1 \n link2 \n link3 ')

urls_text_list = urls_text.split('\n')

#web
ans=' '
ref = 'Me'
st.title('Close Book')
st.caption('\t\t By Ahmed Haytham')
with st.form('LLM'):
    query = st.text_area('Enter query', )
    type_c = st.radio(
    "Type",
    ('Me','Books', 'Urls',),
    horizontal=True,)

    answer = st.form_submit_button('answer')

    if answer:
        if type_c =='Me':
                ans = text_generator.zero_shot(query)

        elif type_c == 'Books':
             #check if the book is already in the database(Embeddings folder) if not gave a msg to the user to embed it
            if (book_name+'.csv') not in os.listdir('Embeddings'):
                st.error('The book is not in the database go to Embedding page to embed it')
            else:
                df = pd.read_csv(os.path.join('Embeddings', book_name+'.csv'))
                ans = text_generator.generate_answer(query, df)
                # read the ref.text file
                ref = open('ref.txt', 'r').read()

        elif type_c == 'Urls':
            df = text_generator.make_urls_df(urls_text_list)
            ans = text_generator.get_genrate_url_answer(query,)
            # read the ref.text file
            ref = open('ref.txt', 'r').read()

    # ans remove first line if it equal ``` and last one if it equal```
        # if ans[0] == '`' and ans[-1] == '`':
            # ans = ans[3:-3]/

        st.markdown(ans)
        if ref != 'Me':
            st.text_area('According to..',ref)
            # st./
more = st.checkbox('More for ref. . .', value=False, key=None)
if more:
     with st.form('more'):
        text = st.text_area('Enter text', )
        type_d = st.radio(
                "Type",
                ('Summary', 'Re-Writing','Explain'),
                horizontal=True,)

        answer = st.form_submit_button('Do')
        if answer:
            if type_d == 'Summary':
                ans = text_generator.Summarize(text)
            if type_d =='Re-Writing':
                ans = text_generator.re_write(text)
            if type_d == 'Explain':
                ans = text_generator.explain(text)

            st.markdown(ans)

more_books = st.checkbox('More for Books. . .', value=False, key=None)
if more_books:
    with st.form('books'):
        summ = st.form_submit_button('Do!')
        if summ:
            #check if the book in summries
            summs = os.listdir('Summrizes')
            if (book_name+'.txt') not in summs:
                text_generator.sumrize_book(book_name)
            # read the text file of book
            ref = open(os.path.join('Summrizes', book_name+'.txt'), 'r').read()
            st.text_area(f'{book_name} summary',ref)
