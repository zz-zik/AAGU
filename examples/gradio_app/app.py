# -*- coding: utf-8 -*-
"""
@Project : AAGU
@FileName: app.py
@Time    : 2025/5/20 下午3:23
@Author  : ZhouFei
@Email   : zhoufei.net@gmail.com
@Desc    : Gradio 创建交互式接口
@Usage   :
"""
import gradio as gr
import socket

# 动态获取主机的 IP 地址
def get_host_ip():
    try:
        host_ip = socket.gethostbyname(socket.gethostname())
    except:
        host_ip = "127.0.0.1"  # 默认回退到本地地址
    return host_ip
