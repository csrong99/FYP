from celery import shared_task

import time

@shared_task
def test():
    print('hi')
    