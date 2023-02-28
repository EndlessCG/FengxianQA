# coding=utf-8
# 回译法
import http.client
import hashlib
import urllib
import random
import json
import time
import pandas as pd

def back_trans(content, lan):
    import http.client
    import hashlib
    import urllib
    import random
    import json
    import time
    import pandas as pd
    appid = '20221111001447059' # 填写你的appid
    secretKey = 'gAoSilzZXNQHLyXoM3zA' # 填写你的密钥
    httpClient = None
    myurl = '/api/trans/vip/translate' #    百度翻译API
    ChineseLang = 'zh' # 原文语种，填写中文 (zh)，也可自动识别 (填auto)
    EnglishLang = lan # 译文语种，填英文 (en)
    httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
   
    # 翻译成英文，from=ChineseLang，to=EnglishLang
    salt = random.randint(32768, 65536)
    sign = appid + content + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    first_url = myurl + '?appid=' + appid + '&q=' + \
    urllib.parse.quote(content) + '&from=' + \
    ChineseLang + '&to=' + EnglishLang + '&salt=' + \
    str(salt) + '&sign=' + sign
    httpClient.request('GET', first_url)
    response = httpClient.getresponse()
    result_all = response.read().decode("utf-8")
    english_result = json.loads(result_all)
    english_result = english_result['trans_result'][0]['dst'] # 获取翻译后的英文文本
    
    # 百度翻译，请求之间要限制频率，等待一秒
    time.sleep(1)
    
    # 翻译回中文，from=EnglishLang，to=ChineseLang
    salt = random.randint(32768, 65536)
    sign = appid + english_result + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    second_url = myurl + '?appid=' + appid + '&q=' + \
    urllib.parse.quote(english_result) + '&from=' + \
    EnglishLang + '&to=' + ChineseLang + '&salt=' + \
    str(salt) + '&sign=' + sign
    httpClient.request('GET', second_url)
    response = httpClient.getresponse()
    result_all = response.read().decode("utf-8")
    chinese_result = json.loads(result_all)
    chinese_result = chinese_result['trans_result'][0]['dst'] # 获取翻译后的中文文本
    return chinese_result

if __name__ == "__main__":
    print(back_trans("Q问答系统是目前应用最广泛的问答系统。这种问答系统的结构框架明了、实现简单、容易理解，非常适合作为问答系统入门学习时的观察对象。针对业务场景中用户最常问、或最有可能问的问题，又称“标准问题”，我们可以提前编制答案，构成问答对。", 'en'))