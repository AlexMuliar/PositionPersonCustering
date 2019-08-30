from db_config import get_conn
from person_split import PersonClustering

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import dask.dataframe as dd
import json
import os, sys
import datetime as dt

conn = get_conn()
CLUSTER_ID = pd.read_sql("SELECT max(cluster_id) as max_id FROM cluster_data", conn)
if CLUSTER_ID.iloc[0].max_id == None:
    CLUSTER_ID = 1
else:
    CLUSTER_ID = CLUSTER_ID.iloc[0].max_id
conn.close()


def get_dress_by_file_id(file_id):
    connection = get_conn()
    df = pd.read_sql(f"\
        SELECT id as object_id, x_center, y_center, height, width, \
                    frame_no, frame_time, dress_features as dress\
        FROM images_data\
        WHERE\
            file_id = { file_id }\
            AND width > 49\
            AND height > 49", connection)
    connection.close()
    df.dress = df.dress.map(lambda x: np.loads(x))
    return df

def cluster(file_id):
    start = dt.datetime.now()
    df = get_dress_by_file_id(file_id)
    if df.shape[0] == 0:
        return 0, -1
    clf = PersonClustering(position_split_alpha=0.4, position_split_min_cluster_size=3, 
            position_split_min_samples=1, time_stretching_degree=6, min_cluster_size_for_cheking_noise=5,
            max_difference_between_person_samples=0.75, noise_filter_compress_n_components=2)
    df = clf.fit_transform(df)
    print(file_id, df.cluster.unique().shape)
    print(f"time: {dt.datetime.now() - start}")
    return df, file_id


def insert_into_db(data, FILE_ID):
    global CLUSTER_ID
    connection = get_conn()
    print('Inserting')
    insert_to_cluster_data = "INSERT INTO `cluster_data`\
         (`file_id`, `cluster_id`, `start_object_image_id`, \
             `end_object_image_id`, `start_object_frame_time`, `end_object_frame_time`, `object_cam_id`, `cluster_age`, `cluster_gender`)\
         VALUES (%i, %i, %i, %i, '%s', '%s', %i, %s, %s)"
    insert_to_file_cluser_object = "INSERT INTO `file_cluster_object`\
        (`object_image_id`, `cluster_data_id`, `file_id`, `object_frame_time`, `object_cam_id`)\
        VALUES (%i, %i, %i, '%s', %i)"
    object_cam_id = pd.read_sql("SELECT cam_id FROM video_files WHERE id = %i" \
                            % FILE_ID, connection).cam_id.tolist()[0]
    for clust_num in data.cluster.unique():
        if clust_num < 0:
            continue
        df = data[data.cluster == clust_num]
        if df.shape[0] == 1:
            continue
        start_object_id = df.object_id.min().tolist()
        end_object_id = df.object_id.max().tolist()
        start_object_frame_time = df[df.object_id == start_object_id].frame_time.tolist()[0]
        end_object_frame_time = df[df.object_id == end_object_id].frame_time.tolist()[0]
        try:
            cluster_age = df[df.age != '0'].age.value_counts().idxmax()
        except:
            cluster_age = '0'
        try:
            cluster_gender = df[df.gender != '0'].gender.value_counts().idxmax()
        except:
            cluster_gender = '0'
        try:
            with connection.cursor() as cursor:
                cursor.execute(insert_to_cluster_data % \
                    (FILE_ID, CLUSTER_ID, start_object_id, end_object_id, \
                        start_object_frame_time, end_object_frame_time, object_cam_id, cluster_age, cluster_gender))
            connection.commit()
            CLUSTER_ID += 1
            with connection.cursor() as cursor:
                cursor.execute("SELECT max(id) as record_id FROM cluster_data")
                record_id = cursor.fetchone()['record_id']
            with connection.cursor() as cursor:
                for i in range(df.shape[0]):
                    # print(insert_to_file_cluser_object % \
                    #     (df.iloc[i].object_id, record_id, FILE_ID, str(df.iloc[i].frame_time)))
                    cursor.execute(insert_to_file_cluser_object % \
                        (df.iloc[i].object_id, record_id, FILE_ID, str(df.iloc[i].frame_time), object_cam_id))
        except Exception as exception:
            print(exception)
        finally:
            pass
    connection.commit()
    connection.close()

def clustering():
    conn = get_conn()
    files = pd.read_sql("SELECT id from video_files", conn).id.tolist()
    conn.close()
    with ProcessPoolExecutor() as executor:
        for df, file_id in executor.map(cluster, files):
            if file_id == -1:
                print('Error')
            else:
                #insert_into_db(df, file_id)
                print(df.shape, file_id)
            
