import json
import sys
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from PIL import Image
import gradio as gr
import requests
#import civitai.py script
from enum import Enum

from modules.api.models import *
from modules.api import api

import os
import requests

root_path = os.getcwd()

folders = {
    "TextualInversion": os.path.join(root_path, "embeddings"),
    "Hypernetwork": os.path.join(root_path, "models", "hypernetworks"),
    "Checkpoint": os.path.join(root_path, "models", "Stable-diffusion"),
    "LORA": os.path.join(root_path, "models", "Lora"),
    "LyCORIS": os.path.join(root_path, "models", "LyCORIS"),
}
url_dict = {
    "modelPage":"https://civitai.com/models/",
    "modelId": "https://civitai.com/api/v1/models/",
    "modelVersionId": "https://civitai.com/api/v1/model-versions/",
    "hash": "https://civitai.com/api/v1/model-versions/by-hash/"
}

def getSubfolders(type):
    folder = folders[type]
    if not folder:
        return
    
    if not os.path.isdir(folder):
        return
    
    prefix_len = len(folder)
    subfolders = []
    for root, dirs, files in os.walk(folder, followlinks=True):
        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            # get subfolder path from it
            subfolder = full_dir_path[prefix_len:]
            subfolders.append(subfolder)
    subfolders.append("/")

    return subfolders
    
def_headers = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}


proxies = None

def get_model_info_by_id(id:str) -> dict:

    if not id:
        return

    r = requests.get(url_dict["modelId"]+str(id), headers=def_headers, proxies=proxies)
    if not r.ok:
        if r.status_code == 404:
            # this is not a civitai model
            return {}
        else:
            return

    # try to get content
    content = None
    try:
        content = r.json()
    except Exception as e:
        return
    
    if not content:
        return 
    
    return content

class IdType(str, Enum):
    modelId = "modelId"
    VersionId = "id"
    

def civitai_api(_: gr.Blocks, app: FastAPI):

    # Download Model From Civitai
    @app.post("/civitai/download/")
    async def civitai_download(id:int,subfolder:str="/",version:int=0):

        model_info = get_model_info_by_id(id)

        # Make sure we have a valid model version
        if version > len(model_info["modelVersions"]):
            version = len(model_info["modelVersions"])-1

        version_id = model_info["modelVersions"][version]["id"]

        # Get the download url
        download_url = model_info["modelVersions"][version]["files"][0]["downloadUrl"]
        folder = folders[model_info["type"]] + subfolder
        filename = model_info["modelVersions"][version]["files"][0]["name"]
        file_path = os.path.join(folder, filename)

        # first request for header

        rh = requests.get(download_url, stream=True, verify=False, headers=def_headers, proxies=proxies)
        total_size = 0
        total_size = int(rh.headers['Content-Length'])
        
        print("Target file path: " + file_path)
        base, ext = os.path.splitext(file_path)

        # check if file is already exist
        count = 2
        new_base = base
        while os.path.isfile(file_path):
            print("Target file already exist.")
            # re-name
            new_base = base + "_" + str(count)
            file_path = new_base + ext
            count += 1

        # use a temp file for downloading
        dl_file_path = new_base+".downloading"


        print(f"Downloading to temp file: {dl_file_path}")

        # check if downloading file is exsited
        downloaded_size = 0
        if os.path.exists(dl_file_path):
            downloaded_size = os.path.getsize(dl_file_path)

        print(f"Downloaded size: {downloaded_size}")

        # create header range
        headers = {'Range': 'bytes=%d-' % downloaded_size}
        headers['User-Agent'] = def_headers['User-Agent']

        # download with header
        r = requests.get(download_url, stream=True, verify=False, headers=headers, proxies=proxies)

        # write to file
        with open(dl_file_path, "ab") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    downloaded_size += len(chunk)
                    f.write(chunk)
                    # force to write to disk
                    f.flush()

                    # progress
                    progress = int(50 * downloaded_size / total_size)
                    sys.stdout.reconfigure(encoding='utf-8')
                    sys.stdout.write("\r[%s%s] %d%%" % ('-' * progress, ' ' * (50 - progress), 100 * downloaded_size / total_size))
                    sys.stdout.flush()

        print()

        # rename file
        os.rename(dl_file_path, file_path)
        print(f"File Downloaded to: {file_path}")

        info_path = file_path.replace(ext, ".civitai.info")
        with open(info_path, 'w') as f:
            f.write(json.dumps(model_info["modelVersions"][version], indent=4))

        return {
            "message": "downloaded",
            "path": file_path,
            "filename": filename,
        }
    
    #get all model subfolders
    @app.get("/civitai/subfolders")
    async def get_subfolders():

        return {
            "LORA": getSubfolders("LORA"),
            "TextualInversion": getSubfolders("TextualInversion"),
            "Hypernetwork": getSubfolders("Hypernetwork"),
            "Checkpoint": getSubfolders("Checkpoint"),
            "LyCORIS": getSubfolders("LyCORIS"),
        }


    

    #check if model is already installed
    @app.get("/civitai/installed")
    async def check_installed(id:int,id_type:IdType=IdType.modelId):
        model_info = get_model_info_by_id(id)

        if not model_info:
            return {
                "installed": False,
            }

        folder = folders[model_info["type"]]

        #loop through all .civitai.info files in folder and its subfolders
        for root, dirs, files in os.walk(folder, followlinks=True):
            for filename in files:
                if filename.endswith(".civitai.info"):
                    with open(os.path.join(root, filename), 'r') as f:
                        try:
                            model_info = json.load(f)
                            if model_info[id_type] == id:
                                return {
                                    "installed": True,
                                    "filename": filename.replace(".civitai.info","")
                                }
                        except Exception as e:
                            print(e)
                            return {
                                "installed": False,
                            }
        



try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(civitai_api)
except:
    pass