import requests

url = 'http://localhost:5000/predict'
#url='http://127.0.0.1:12344/predict'
ok={
		"impression_id":"c4ca4238a0b923820dcc509a6f75849b", "impression_time":4,"user_id":3,
		"app_code":4, "os_version":"old","is_4G":5,"is_click":5,"server_time":6, "device_type":"android", "session_id":7,
       "item_id":4, "item_price":4, "category_1":5, "category_2":5, "category_3":5,"product_type":5
	}

r = requests.post(url,json=ok)

print(r.json())
