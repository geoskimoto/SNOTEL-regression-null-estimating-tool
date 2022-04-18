# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:25:35 2021

@author: Beau.Uriona
"""

import sqlite3
from sqlite3 import Error as SQLError
from os import path, getenv
import pandas as pd

DB_DIR = path.dirname(path.realpath(__file__))
API_SERVER = getenv("API_SERVER", "https://api.snowdata.info")
NETWORKS = ("SNTL", "SNTLT")


def get_meta(networks=NETWORKS, api_domain=API_SERVER):
    dfs = []
    orient = "records"
    for network in networks:
        print(f"  Getting meta for {network} sites...")
        endpoint = "stations/getMeta"
        args = f"?site_id=ALL&network={network}&format=json&orient={orient}"
        url = f"{api_domain}/{endpoint}{args}"
        df = pd.read_json(url, orient=orient)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def parse_label(row):
    return f"{row['name']} ({row['station_id']}) ({row['elevation']}')"


def parse_triplet(triplet, index):
    trip_arr = triplet.split(":")
    return trip_arr[index]


def format_meta(df):
    df["station_id"] = df["stationTriplet"].apply(lambda x: parse_triplet(x, 0))
    df["state"] = df["stationTriplet"].apply(lambda x: parse_triplet(x, 1))
    df["network"] = df["stationTriplet"].apply(lambda x: parse_triplet(x, 2))
    df["label"] = df.apply(parse_label, axis=1)
    df.rename(columns={"stationTriplet": "triplet"}, inplace=True)
    dtype_dict = {
        "name": str,
        "label": str,
        "station_id": int,
        "state": str,
        "network": str,
        "triplet": str,
        "elevation": int,
        "latitude": float,
        "longitude": float,
        "huc": str,
        "beginDate": str,
    }
    df = df[dtype_dict.keys()].astype(dtype_dict)
    return df


def convert_to_sqlite(df, db_path=path.join(DB_DIR, "meta.db")):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df.to_sql(name="meta", con=conn, if_exists="replace")
    except SQLError as e:
        print(f"Error converting metadata to sqlite - {e}")
    finally:
        if conn:
            conn.close()


def refresh_meta():
    print("Refreshing metadata/site list database...")
    df = format_meta(get_meta()).sort_values(by="name", axis=0)
    csv_path = path.join(DB_DIR, "meta.csv")
    df.to_csv(csv_path, index=False)
    db_path = path.join(DB_DIR, "meta.db")
    convert_to_sqlite(df, db_path)
    print("  \nSuccess!")


if __name__ == "__main__":
    refresh_meta()
