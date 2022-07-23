import time
import requests
import threading


class TestThread(threading.Thread):

    log_mutex = threading.Lock()

    def __init__(self, url, index):
        super().__init__()
        self.url = url
        self.index = index
        self.result = None

    def run(self) -> None:
        begin = time.time()
        self.result = requests.get(self.url)
        end = time.time()
        TestThread.log_mutex.acquire()
        print('completed:', self.index, self.result.status_code, 'elapsed:', round(end - begin, 2))
        TestThread.log_mutex.release()


def pressure_test_url(url: str, instances: int):
    """pressure test of apis"""
    threads = []
    for idx in range(instances):
        threads.append(TestThread(url, idx))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if all(thread.result.status_code == 200 for thread in threads):
        print(f'Test Url {url} pressure {instances} PASSED!')
    else:
        print(f'Test Url {url} pressure {instances} FAILED!')


def pressure_test_urls(url_instances: list):
    """pressure test of apis"""
    threads, idx = [], 0
    for url, instances in url_instances:
        for _ in range(instances):
            threads.append(TestThread(url, idx))
            idx += 1
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if all(thread.result.status_code == 200 for thread in threads):
        print(f'Test PASSED!')
    else:
        print(f'Test FAILED!')


def main():
    url_search_ent = 'http://166.111.5.105:17201/sykb/api/search_ent?key=aspirin&pageIndex=1&pageSize=40'
    pressure_test_url(url_search_ent, 20)
    url_search = 'http://166.111.5.105:17201/sykb/api/search?head=aspirin&tail=cough&pageIndex=1&pageSize=10'
    pressure_test_url(url_search, 6)
    url_doc_info = 'http://166.111.5.105:17201/sykb/api/docinfo?pmid=22934008'
    pressure_test_url(url_doc_info, 20)
    url_ent_info = 'http://166.111.5.105:17201/sykb/api/entinfo?meshid=D001986'
    pressure_test_url(url_ent_info, 20)
    pressure_test_urls([(url_search_ent, 20), (url_search, 6), (url_doc_info, 20), (url_ent_info, 20)])


if __name__ == '__main__':
    main()
