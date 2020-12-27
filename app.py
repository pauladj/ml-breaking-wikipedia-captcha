import SessionState
import numpy as np
import streamlit as st
from imutils import paths
from streamlit.script_request_queue import RerunData
from streamlit.script_runner import RerunException
from test_model import import_model
from test_model import predict_captcha_image

title = "Breaking Wikipedia's captcha 🔠"
st.title(title)

st.markdown('<style>{}</style>'.format(""".block-container { 
                                            text-align: center;
                                            }"""), unsafe_allow_html=True)


@st.cache
def get_captcha_list():
    # Always get the same random images
    np.random.seed(2)
    # randomly sample a 20 of the input images
    image_paths = list(paths.list_images('./downloads'))
    image_paths = np.random.choice(image_paths, size=(20,),
                                   replace=False)
    return image_paths


@st.cache
def get_random_captcha(captchas, captcha_number):
    num_captchas = len(captchas)
    captcha = captchas[np.random.randint(0, num_captchas, 1)[0]]
    return captcha


def run_labeler():
    state = SessionState.get(captcha_number=0)
    captcha_image_list = get_captcha_list()


    captcha = get_random_captcha(captcha_image_list, state.captcha_number)
    st.image(captcha)

    col_1, col_2 = st.beta_columns(2)
    if col_1.button('Break it!'):
        model, lb = import_model('output')
        image, text = predict_captcha_image(captcha, model, lb)
        st.image(image)
        st.write(text)

    if col_2.button('Next captcha'):
        state.captcha_number += 1
        raise RerunException(RerunData(widget_states=None))


# main block run by code
if __name__ == '__main__':
    run_labeler()
