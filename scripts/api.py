import asyncio
import json
import sys
from addict import Dict
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
from aiofiles import open as aio_open
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
    subfolders.append("/")
    for root, dirs, files in os.walk(folder, followlinks=True):
        for dir in dirs:
            full_dir_path = os.path.join(root, dir)
            # get subfolder path from it
            subfolder = full_dir_path[prefix_len:]
            subfolders.append(subfolder)

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
    
# Create a list of all installed models beforehand
installed_models_list = []
is_downloading  = False

def civitai_api(_: gr.Blocks, app: FastAPI):

    # Download Model From Civitai
    @app.post("/civitai/download/")
    async def civitai_download(id:int,subfolder:str="/",version:int=0,image:int=0):


            


        model_info = get_model_info_by_id(id)

        # Make sure we have a valid model version
        if version > len(model_info["modelVersions"]):
            version = len(model_info["modelVersions"])-1

        version_id = model_info["modelVersions"][version]["id"]

        print("Downloading Model: " + model_info["name"])    

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

        print("")

        # rename file
        os.rename(dl_file_path, file_path)
        print(f"File Downloaded to: {file_path}")
        is_downloading = False

        # Save the model info to a .civitai.info file
        info_path = file_path.replace(ext, ".civitai.info")
        with open(info_path, 'w') as f:
            f.write(json.dumps(model_info["modelVersions"][version], indent=4))

        # Download image from imageUrl
        image_path = file_path.replace(ext, ".preview.png")
        image_url = model_info["modelVersions"][version]["images"][image]["url"]
        image_r = requests.get(image_url, stream=True, verify=False, headers=def_headers, proxies=proxies)
        with open(image_path, "wb") as f:
            f.write(image_r.content)
            





        refresh_installed_models()
        return {
            "message": "downloaded",
            "path": file_path,
            "filename": filename.replace(ext, ""),
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


    @app.post("/civitai/refresh-installed")
    async def refresh_installed():
        refresh_installed_models()
        return {
            "message": "refreshed"
        }

    @app.get("/civitai/installed")
    async def check_installed(id: int, id_type: IdType = IdType.modelId):
        model_info = get_model_info_by_id(id)

        if not model_info:
            return {
                "installed": False,
            }

        folder = folders[model_info["type"]]

        async def search_in_files(root):
            for filename in os.listdir(root):
                if filename.endswith(".civitai.info"):
                    file_path = os.path.join(root, filename)
                    try:
                        async with aio_open(file_path, 'r') as f:
                            content = await f.read()
                            model_info = json.loads(content)
                            if model_info[id_type] == id:
                                return {
                                    "installed": True,
                                    "filename": filename.replace(".civitai.info", "")
                                }
                    except Exception as e:
                        print(e)
        
        for root, dirs, files in os.walk(folder, followlinks=True):
            result = await search_in_files(root)
            if result:
                return result
        
        return {
            "installed": False,
        }
        
    
    @app.post("/civitai/installed-multiple")
    async def check_installed_multiple(payload: Dict[str, list[int]]):
        ids = payload.get("ids", [])
        installed_models = {}

        # Initialize the installed_models dictionary with all IDs
        # for id in ids:
        #     installed_models[str(id)] = {
        #         "installed": False,
        #     }

        # Check each ID against the pre-created list of installed models
        for id in ids:
            for model_info in installed_models_list:
                if model_info["modelId"]:
                    if model_info["modelId"] == id:
                        installed_models[str(id)] = {
                            "installed": True,
                            "filename": model_info["filename"].replace(".civitai.info", "")
                        }
                        break

        return installed_models


def refresh_installed_models():
    installed_models_list.clear()
    for root, dirs, files in os.walk(os.path.join(root_path, "models"), followlinks=True):
        for filename in files:
            if filename.endswith(".civitai.info"):
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        model_info = json.loads(content)
                        model_info["filename"] = filename
                        if model_info["modelId"]:
                            installed_models_list.append(model_info)
                # except Exception as e:
                #     print(e)

refresh_installed_models()

try:
    import modules.script_callbacks as script_callbacks

    script_callbacks.on_app_started(civitai_api)
except:
    pass
