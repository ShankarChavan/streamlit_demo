import streamlit as st
from multiapp import MultiApp
from apps import subscription,pencilsketch # import your app modules here

app = MultiApp()

# Add all your application here
app.add_app("Subscription Model", subscription.app)
app.add_app("Pencil Sketch", pencilsketch.app)

# The main app
app.run()


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="http://shankarchavan.github.io" target="_blank">Shankar Chavan</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
