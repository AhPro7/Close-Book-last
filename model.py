import os
import google.generativeai as palm
from tqdm import tqdm
palm.configure(api_key=os.environ["GOOGLE_API_KEY"])
import pandas as pd

import google.generativeai as palm

import chromadb
from chromadb.api.types import Documents, Embeddings

from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader  #load pdf
from langchain.indexes import VectorstoreIndexCreator #vectorize db index with chromadb
from langchain.text_splitter import CharacterTextSplitter #text splitter
import numpy as np

import ast
import textwrap

from langchain.document_loaders import UnstructuredURLLoader  #load urls into docoument-loader

#models

# embedText
models = [m for m in palm.list_models() if 'embedText' in m.supported_generation_methods]
emb_model = models[0]

#Generate text
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
text_model = models[0].name

class TextGenerator:
    df_d = None
    def __init__(self, text_model=text_model,emb_model=emb_model):
        self.text_model = text_model
        self.emb_model = emb_model

    def find_best_passage(self,query, dataframe):
        """
        Compute the distances between the query and each document in the dataframe
        using the dot product.
        """
        query_embedding = palm.generate_embeddings(model=self.emb_model, text=query)
        df_d = dataframe

        dataframe['Embeddings'] = dataframe['Embeddings'].apply(ast.literal_eval)

        dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding['embedding'])
        idx = np.argmax(dot_products)
        
        text = dataframe.iloc[idx]['Text']
        # save the text in ref.text file
        with open('ref.txt', 'w') as f:
            f.write(text)
        return dataframe.iloc[idx]['Text'],idx



    
    def find_best_3_passage(self,query, dataframe):
        """
        we will return 3 best passages
        we will use the previous function and then remove the best passage from the dataframe 3 times
        """
        best_passages = []
        idxs = []
        dataframe['Embeddings'] = dataframe['Embeddings'].apply(ast.literal_eval)

        for i in range(3):
            best_passage,idx = self.find_best_passage(query, dataframe)
            idxs.append(idx)
            best_passages.append(best_passage)
            dataframe = dataframe[dataframe['Text'] != best_passage]
        return best_passages
       

    def make_prompt(self,query, relevant_passage):
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = textwrap.dedent("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
        Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and converstional tone and gave example. \
        If the passage is irrelevant to the answer, you may ignore it.
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

            ANSWER:
        """).format(query=query, relevant_passage=escaped)

        return prompt
    
    
    def generate_answer(self,query, dataframe):
        """
        generate the answer of the query
        """
        res = palm.generate_text(prompt=self.make_prompt(query, self.find_best_passage(query, dataframe)[0]),
                                temperature=0.3,
                                max_output_tokens=2000

                                 )

        return res.result

    def generate_3_answers(self,query, dataframe):
        """
        generate the answer of the query
        """
        res=[]
        for i in range(3):
            re = palm.generate_text(prompt=self.make_prompt(query, self.find_best_3_passage(query, dataframe)[i]),)
            res.append(re.result)
        return res
    
    def zero_shot(self,prompt):
        prompt = 'change your name to "Close-Book"(based on PALM2) ,you are an AI chat tool to help CS students in ther studies'+prompt
        res = palm.generate_text(prompt=prompt,
                                temperature=0.3,
                                max_output_tokens=2000
                                 )
        return res.result

    def make_urls_df(self,urls):

        def make_embeddings(text):
            return palm.generate_embeddings(model=emb_model,text=text)['embedding']

        """
        make a dataframe of urls
        """
        loader = UnstructuredURLLoader(urls=urls)
        texts = loader.load()
        text_list = []
        for text in texts:
            text_list.append(text.page_content)
        splitter = CharacterTextSplitter(chunk_size=1500)
        pargraphs = splitter.create_documents(text_list)
        pargraphs_text=[]
        for pra in pargraphs:
            pargraphs_text.append(pra.page_content)
        text = {'Text':pargraphs_text}
        df = pd.DataFrame(text)
        df['Embeddings'] = df['Text'].apply(make_embeddings)
        # save the df in hidden file calles .urls.csv
        df.to_csv('.urls.csv')
        return df
    
    def  get_genrate_url_answer(self,query,name='.urls.csv'):
        """
        generate the answer of the query
        """
        df = pd.read_csv(name)

        res = palm.generate_text(prompt=self.make_prompt(query, self.find_best_passage(query, df)[0]),
                                temperature=0.3,
                                max_output_tokens=2000)
        # print(res.result)
        return res.result
    
    def Summarize(self,prompt):
        prompt = 'Summarize this text:'+prompt
        res = palm.generate_text(prompt=prompt,
                                temperature=0.3,
                                max_output_tokens=2000
                                 )
        return res.result
    
    def re_write(self,prompt):
        prompt = 'Rewrite the following paragraph to make it more concise and engaging. Use your own words and style.:'+prompt
        res = palm.generate_text(prompt=prompt,
                                temperature=0.3,
                                max_output_tokens=2000
                                 )
        return res.result
    
    def explain(self,prompt):
        prompt = """
                Explain the following paragraph in your own words, and make sure to answer the following questions:
                * What is the main topic of the paragraph?
                * What are the key points of the paragraph?
                * How does the paragraph support its main points?
                Use clear and concise language.
                """+prompt
        
        res = palm.generate_text(prompt=prompt,
                                temperature=0.3,
                                max_output_tokens=2000
                                 )
        
        return res.result
    
    def sumrize_book(self,book_name):
        df = pd.read_csv('Embeddings/'+book_name+'.csv')
        summrizes = []
        for i in df['Text']:
            summrizes.append(self.Summarize(i))
            summrizes.append('____________________________________')

            # save the arry in file in folder file name is book_name.txt
        with open('Summrizes/'+book_name+'.txt', 'w') as f:
            for item in summrizes:
                f.write("%s\n" % item)