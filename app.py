# By Grant Espe
# Streamlit movie recommender App

import streamlit as st
import pandas as pd
import numpy as np



def myIBCF(newuser): # takes a 100 value row of a user's ratings
    # fill nas
    S = st.session_state.similarity
    filled_S = S.fillna(0)
    filled_w = newuser.copy()
    newuser = newuser.replace(0, np.nan, inplace=False)
    #filled_w = newuser.copy().fillna(0)

    preds = (filled_S @ filled_w.T.values) 
    preds = preds / (filled_S @ ((np.logical_not(newuser.T.isna()))).values)

    # fix indexing 
    preds.index = [i for i in range(0,100)]
    # ignore already rated movies
    preds = preds[filled_w.T == 0].stack().nlargest(10)
    # convert back to movie indexes
    #preds = preds.rename(index=lambda x: f'm{x+1}')
    preds = preds.reset_index(level=1, drop=True) # fix weird pandas thing

    return preds



def update_rating(i):
    # get the value of the slider
    key = f"rating_{i}"

    # update it in the dataframe
    st.session_state.movies.loc[i, 'rating'] = st.session_state[f"rating_{i}"]
    print(st.session_state.movies.loc[i])

def image_url(image_id):
    return f"https://liangfgithub.github.io/MovieImages/{image_id}.jpg"

def display_grid():

    # display the grid
    idx = 0 
    movie_ids = st.session_state.movies.MovieID.values.tolist()
    titles = st.session_state.movies.Title.values.tolist()

    # 4 column grid
    cols = st.columns(4) 
    while idx < len(movie_ids):
        # row by row
        for d in range(4):
            cols[d].image(image_url(movie_ids[idx]), width=150, caption=titles[idx])
            cols[d].select_slider(
                            f"Rate Movie {idx+1}", 
                            options=[0, 1, 2, 3, 4, 5],
                            value=0,
                            key=f"rating_{idx}",
                            on_change=update_rating,
                            args=(idx,),
                            label_visibility="collapsed"
                        )
            #add space before next image
            cols[d].text("")
            cols[d].text("")
            cols[d].text("")
            cols[d].text("")

            idx += 1

# get the 100 ratings, and use IBCF to get top 10 recommendations
def get_recommendations():
    my_ratings = st.session_state.movies['rating']
    my_ratings = pd.DataFrame([my_ratings.values.flatten()])
    # get the top 10 predictions
    preds = myIBCF(my_ratings)

    # grab that info from the movies
    selected_movies = preds.index.tolist()
    movies = st.session_state.movies
    my_reccs = movies.loc[selected_movies]

    st.session_state.selected_movies = selected_movies
    st.session_state.my_reccs = my_reccs
    print(selected_movies)
    print(my_reccs)


st.title('Movie Recommender - Grant Espe')

if 'movies' not in st.session_state:
    # just get the first 100 movies
    #movies = pd.read_csv('movies.dat', sep='::', 
    movies = pd.read_csv('https://github.com/gtespe/moviereccappfiles/raw/refs/heads/main/movies.dat', sep='::', 
                        engine = 'python', 
                        encoding="ISO-8859-1", 
                        header = None,
                        nrows=100)

    movies.columns = ['MovieID', 'Title', 'Genres']
    # placeholder ratings columns
    movies['rating'] = 0
    st.session_state.movies = movies

if 'similarity' not in st.session_state:
    # load 100 x 100 similarity matrix
    #st.session_state.similarity = pd.read_csv('similarity.csv', index_col=0)
    st.session_state.similarity = pd.read_csv('https://raw.githubusercontent.com/gtespe/moviereccappfiles/refs/heads/main/similarity.csv', index_col=0)

display_grid()

# setup sidebar for generating recommendation
with st.sidebar:
    st.text("Rate some movies by adjusting the sliders, then click the button!")
    st.button("Generate Recommendations",
                key='generate_button',
                on_click=get_recommendations)
    if 'selected_movies' in st.session_state:
        # row by row
        if len(st.session_state.selected_movies) == 0:
            st.text("Select some movies first!")

        else:
            st.text("Your top 10 recommendations:")
            for i in st.session_state.selected_movies:
                st.image(image_url(i+1), width=100, caption=st.session_state.my_reccs['Title'][i])
