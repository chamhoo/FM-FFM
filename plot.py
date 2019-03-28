import psutil
import os


def memory(string):
    info = psutil.virtual_memory()
    print(string, psutil.Process(os.getpid()).memory_info().rss/10**9, ' / ', info.total/10**9)
