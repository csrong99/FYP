import firebase_admin
from firebase_admin import credentials
import numpy
import urllib
import pyrebase
from datetime import datetime, timedelta
from threading import Timer
from django import template
from django_cron import CronJobBase, Schedule
register = template.Library()

FIREBASE_CONFIG = {
    "apiKey": "AIzaSyDEStvuQpFoqJEBwYp5MdvfEIAqUGvyFMM",
    "authDomain": "fyp-aloe-vera.firebaseapp.com",
    "databaseURL": "https://fyp-aloe-vera.firebaseio.com",
    "projectId": "fyp-aloe-vera",
    "storageBucket": "fyp-aloe-vera.appspot.com",
    "messagingSenderId": "728231515531",
    "appId": "1:728231515531:web:31d5c6705ad2da03a04b8e",
    "measurementId": "G-WZH1924DPJ"
}


def firebase_initialization():
    firebase = pyrebase.initialize_app(FIREBASE_CONFIG)
    return firebase


@register.simple_tag
def write_database(data):
    firebase = firebase_initialization()
    num_children = get_children_count()+1
    av_id = "av{:02}".format(num_children)
    firebase.database().child('aloevera').child(av_id).set(data)


@register.simple_tag
def update_database(data, node):
    firebase = firebase_initialization()
    firebase.database().child('aloevera').child(node).update(data)


@register.simple_tag
def get_aloe_vera(id):
    firebase = firebase_initialization()
    av = firebase.database().child('aloevera').child(id).get()
    return av.val()


@register.simple_tag
def get_aloe_vera_history(avid, hid):
    firebase = firebase_initialization()
    av = firebase.database().child('aloevera').child(
        avid).child('histories').child(hid).get()
    return av.val()


@register.simple_tag
def update_aloe_vera(id, data):
    firebase = firebase_initialization()
    av = get_aloe_vera(id)
    av_dic = {
        'condition': av['condition'],
        'datetime': av['datetime'],
        'height': av['height'],
        'width': av['width']
    }

    num_histories = get_history_count(id) + 1
    hid = "h{:02}".format(num_histories)
    firebase.database().child('aloevera').child(id).update(data)
    firebase.database().child('aloevera').child(
        id).child('histories').child(hid).set(av_dic)


@register.simple_tag
def get_children_count():
    firebase = firebase_initialization()
    num = len(firebase.database().child('aloevera').get().val())
    return num


@register.simple_tag
def get_history_count(id):
    firebase = firebase_initialization()
    num_histories = len(firebase.database()
                        .child('aloevera')
                        .child(id)
                        .get()
                        .val()['histories'])
    return num_histories


def get_condition_days(id):
    firebase = firebase_initialization()
    histories = firebase.database().child(
        'aloevera').child(id).get().val()['histories']
    cond_history = [histories[i]['condition'] for i in list(histories.keys())]
    latest_cond = cond_history[-1]
    uniq_cond_his = []
    for i in list(histories.keys()):
        if histories[i]['condition'] == latest_cond:
            uniq_cond_his.append(histories[i]['datetime'])
    start_datetime = datetime.strptime(uniq_cond_his[0], "%d/%m/%Y %H:%M:%S")
    # end_datetime = datetime.strptime(uniq_cond_his[-1], "%d/%m/%Y %H:%M:%S")
    # duration = end_datetime - start_datetime
    return latest_cond, start_datetime


@register.simple_tag
def to_str(value):
    return str(value)
