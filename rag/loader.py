import os
from common.config import LAW_API_KEY
import requests
import pandas as pd

# 법령 MST번호 찾을 때 사용
'''
url = "https://www.law.go.kr/DRF/lawSearch.do"

params = {
    "OC": LAW_API_KEY,
    "target": "law",
    "type": "XML",
    "query": "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법"
}
'''

def data_load():
    url = "https://www.law.go.kr/DRF/lawService.do"

    params = {
        "OC": LAW_API_KEY,
        "target": "law",
        "MST": "288781",
        "type": "XML",
        "efYd": "20260820"
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    return response
