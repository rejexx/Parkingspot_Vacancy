# Modified from: https://github.com/thoppe/streamlit-skyAR
# who got it from: https://stackoverflow.com/a/39225039
import requests
import streamlit as st # only if you are using show_progress_bar...

def download_file_from_google_drive(id, destination, show_progress_bar=False):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination, show_progress_bar)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination, show_progress_bar):
    CHUNK_SIZE = 32768

    #Make a progress bar
    if show_progress_bar:
        prog_bar = st.progress(0)
        size_so_far = 0
        total_size = 257557808 # would be good to make this dynamic later

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                if show_progress_bar:
                    size_so_far += CHUNK_SIZE
                    # min makes sure progress bar doesn't exceed 1
                    prog_bar.progress(min(size_so_far/total_size, 1))

    if show_progress_bar:
        prog_bar.empty()