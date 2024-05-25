from datetime import datetime

datetime_str = '2022/3/16 13:55:26'

datetime_object = datetime.strptime(datetime_str, '%y/%m/%d %H')

print(type(datetime_object))
print(datetime_object)  # printed in default format

