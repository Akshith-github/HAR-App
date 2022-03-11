# command to run streamlit: streamlit run app.py


import streamlit as st #import streamlit
# import st_player
# from streamlit_player  import st_player

# enable wide screen
st.set_page_config(layout='wide')

# Title

st.title(">>> Human Activity Recognition 🤾‍♂️🚶‍♂️🏂")

# modes 
# radio button to select mode of video input
mode = st.radio(" # Input Feed : ",["VideoFile","Video Camera Feed"],0, )

# Header
l,m,r = st.columns([1,3,1])

m.write("---")
m.header("This is a simple web app to predict human activity")
# st.markdown("![Alt Text](C:/Users/akshi/Desktop/HAR/template.gif)",unsafe_allow_html=True)
m.image("./template.gif")
m.write("---")


# "---"
# Subheader
c1, c_, c2 = st.columns([3, 1, 4])
c1.subheader("1) To start, please select a model")

modelName = c1.selectbox("Select a model", ["LSTM + Detectron2"])
#["SVM", "KNN", "Random Forest", "Logistic Regression", "Naive Bayes", "Decision Tree"])

if modelName:
    with c1:
        with st.spinner("Loading model..."):
            # st.write("Loading model...")
            from HARmodels.model import loadModel
            # loadModel(modelName)
            # sleep 2 seconds
            from time import sleep
            sleep(2)
            model = loadModel(modelName)
            c1.write("Model loaded")
# "---"

# Subheader
c2.subheader("2) Upload the video file")

file = c2.file_uploader("Upload a video file", 
        type=["mp4", "avi", "mov", "mkv", "wmv", "flv", "mpg", "mpeg"])

# "---\n---"
"---"
# Subheader for the video stream display
l,y,r = st.columns([9,1,9])

l.subheader("3) Display Actual video stream")
if file: 
    l.video(file)

r.subheader("4) Display Predicted video stream")
if file:
    with r:
        with st.spinner("Predicting..."):
            # st.write("Predicting...")
            from HARmodels.model import predict
            from time import sleep
            sleep(2)
            predictedVideo = predict(model, file)
            r.video(predictedVideo)
    # r.video(file)
