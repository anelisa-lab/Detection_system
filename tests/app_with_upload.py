"""AppTest harness: runs app.py with the file uploader returning the file named in UPLOAD_FILES."""
import os
import streamlit as st


class _F:
    def __init__(self, path):
        self.path, self.name = path, os.path.basename(path)

    def getvalue(self):
        return open(self.path, "rb").read()


st.file_uploader = lambda *a, **k: [_F(p) for p in os.environ["UPLOAD_FILES"].split("|")]
exec(open(os.path.join(os.path.dirname(__file__), "..", "app.py"), encoding="utf8").read())
