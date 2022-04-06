#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
=================================================
@Project -> File   ：thunlp_demo_backend -> __init__.py.py
@IDE    ：PyCharm
@Author ：zhoupeng@mail.tsinghua.edu.cn
@Date   ：2022/4/1 5:28 PM
@Desc   ：
==================================================
Version:
    NO          Date            TODO        Author
    V1.1.0      2022/4/1 5:28 PM             zhoupeng@mail.tsinghua.edu.cn
"""
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app, supports_credentials=True)


def initialize(routed=True):
    if routed:
        from . import app, sykb_route
