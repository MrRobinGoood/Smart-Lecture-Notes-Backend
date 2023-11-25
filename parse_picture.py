from icrawler.builtin import  GoogleImageCrawler
import re

dir_name = 'resources/pic'

google_crawler = GoogleImageCrawler(storage={'root_dir' : dir_name})

filters = dict(
    size='medium',
    color='color',
    license='noncommercial,modify',
    date=((2018, 1, 1), (2023, 11, 23))
)

# all_urls = []
substring = "image #"

def checkURL(log_input):
    line = log_input.getMessage()
    res = re.search(substring, line)
    if res:
        line = line.split()
        line = line[2]
        print(line)
        return line
        # all_urls.append(line)
        # print(all_urls)
    # print(all_urls)
def imageCrawler(query, num, filters):
    google_crawler.downloader.logger.addFilter(checkURL)
    google_crawler.crawl(keyword=query, max_num=num, filters= filters)

imageCrawler("Описание задачи", 7, filters)

