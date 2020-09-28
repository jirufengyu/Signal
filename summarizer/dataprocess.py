import re
import os
import pandas as pd

class Sougou(object):
    def split_language_database(self):
        file="news_sohusite_xml.dat"
        text = open(file, 'rb').read().decode("gb18030")
        # 匹配 url 和 正文内容
        content = re.findall('<url>(.*?)</url>.*?<contenttitle>(.*?)</contenttitle>.*?<content>(.*?)</content>',
                             text, re.S)
        df = pd.DataFrame(columns=('title','content'))
        # 根据 url 存放每一个 正文内容
        for news in content:
            url_title = news[0]
            content_title = news[1]
            news_text = news[2]
            # 提取正文的类别
            #title = re.findall('http://(.*?).sohu.com', url_title)[0]
            # 存储正文
            if len(content_title) > 0 and len(news_text) > 30:
                df=df.append({'title':content_title,'content':news_text},ignore_index=True)
        df.to_csv("sgnewfull.csv",sep='\t')
if __name__ == '__main__':
    sg = Sougou()
    sg.split_language_database()
    dt=pd.read_csv("sgnewfull.csv",sep='\t',skiprows=1,names=['title','content'])
    print(dt.head())
    