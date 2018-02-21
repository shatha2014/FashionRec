import json
"""
data = []
with open('data/append_70k/total.json') as f:
    for line in f:
        data.append(json.loads(line))
print len(data)
print data[0]
with open('data/append_70k/total2.json', 'w') as outfile:
    json.dump(data, outfile)
"""

def is_json(myjson):
    try:
        json_object = json.loads(myjson)
    except ValueError, e:
        return False
    return True

with open('data/append_70k/total2.json') as f:
    jsonstr = f.read()
    print(is_json(jsonstr))