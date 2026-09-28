import streamlit.web.cli as stcli

import sys

import os



# Importamos 'app' e 'utils' para garantir que o PyInstaller

# os "veja" e os inclua no .exe

import app

import utils

import pdfplumber

import sqlite3



def get_app_path():

    # Retorna o caminho para o 'app.py'

    if getattr(sys, 'frozen', False):

        return os.path.join(sys._MEIPASS, 'app.py')

    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.py')



if __name__ == "__main__":

    app_path = get_app_path()

   

    sys.argv = [

        "streamlit",

        "run",

        app_path,

        "--global.developmentMode", "false",  # <-- ADICIONADO: Desliga o modo dev

        "--server.port", "8501",            

        "--server.headless", "true",        

        "--server.runOnSave", "false",

        "--browser.gatherUsageStats", "false"

    ]

   

    sys.exit(stcli.main())

