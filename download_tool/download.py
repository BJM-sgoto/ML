import flickrapi
import urllib
from PIL import Image
import time
import os
import sys

# Flickr api access key 
flickr=flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)


keyword = sys.argv[1]
year = sys.argv[2]

photos = flickr.walk(text=keyword,
	tag_mode='all',
	tags=keyword,
	extras='url_z',
	per_page=100,					 # may be you can try different numbers..
	sort='relevance',
	min_upload_date= year + '-01-01',
	max_upload_date= year + '-12-31')
files = os.listdir('./output/' + keyword + '/')
files.sort()
count = 0
if len(files)!=0:
	count = int(files[-1][:-4])
for i, photo in enumerate(photos):
	url = photo.get('url_z')
	print(i, url)
	if url!=None:
		# Download image from the url and save it to '00001.jpg'
		urllib.request.urlretrieve(url, './output/' + keyword + '/{:06d}.jpg'.format(count))
		time.sleep(1)
		count += 1
