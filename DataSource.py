# -*- coding: utf-8 -*-
# 数据源

import pandas as pd
import numpy as np

import yfinance as yf
import os

# 如需通过代理访问网络，请取消以下注释并填入你的代理地址
# proxy_url = 'http://127.0.0.1:7890'
# os.environ['HTTP_PROXY'] = proxy_url
# os.environ['HTTPS_PROXY'] = proxy_url

sp500 = yf.download('^GSPC', start='2025-01-01', end='2026-01-01', auto_adjust=True)
print(sp500)
sp500.to_csv('YahooFinance.csv')