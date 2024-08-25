#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import praw
import pandas as pd
import pytz
from datetime import datetime

#HW4

from utils import *
the_path = r"your path"
the_path1 = r"output path"

import numpy as np

#end of HW4#

subreddit_channel = 'politics'

reddit = praw.Reddit(
     client_id="0zE7_AAXIely_3a37Gho3w",
     client_secret="HUGNgNN9gV3zt9g34IIfhryQxaPZ2Q",
     user_agent="testscript by u/fakebot3",
     username="your username",
     password="*your password",
     check_for_async=False
 )

print(reddit.read_only)

def conv_time(var):
    tmp_df = pd.DataFrame()
    tmp_df = tmp_df.append(
        {'created_at': var},ignore_index=True)
    tmp_df.created_at = pd.to_datetime(
        tmp_df.created_at, unit='s').dt.tz_localize(
            'utc').dt.tz_convert('US/Eastern') 
    return datetime.fromtimestamp(var).astimezone(pytz.utc)

def get_reddit_data(var_in):
    import pandas as pd
    tmp_dict = pd.DataFrame()
    tmp_time = None
    try:
        tmp_dict = tmp_dict.append({"created_at": conv_time(
                                        var_in.created_utc)},
                                    ignore_index=True)
        tmp_time = tmp_dict.created_at[0] 
    except:
        print ("ERROR")
        pass
    tmp_dict = {'msg_id': str(var_in.id),
                'author': str(var_in.author),
                'body': var_in.body, 'datetime': tmp_time}
    return tmp_dict

#HW4# 

for comment in reddit.subreddit(subreddit_channel).stream.comments():
    tmp_df = get_reddit_data(comment)
    
    redditcomment = clean_text(tmp_df["body"])
    redditcomment = rem_sw(redditcomment)
    redditcomment = my_stem(redditcomment)
    
    my_vec = read_pickle(the_path, 'vectorizer')
    redditcomment_vec = my_vec.transform([redditcomment]).toarray()
    
    pca = read_pickle(the_path, 'pca')
    redditcomment_pca = pca.transform(redditcomment_vec)
    
    my_model_fun = read_pickle(the_path, 'my_model')
    the_pred = my_model_fun.predict(redditcomment_pca)
    the_conf = my_model_fun.predict_proba(redditcomment_pca)

    print (tmp_df["body"], the_pred, np.max(the_conf))

    
#end of HW4#

