from collections import OrderedDict
import hashlib
import base64
import json
from time import time, sleep


def websocket_signature(key, secret, action, data):
    tsp = int(time() * 1000)
    msg = {'_': tsp,'_ackey': key,'_acsec': secret,'_action': action}
    msg.update(data)
    s_msg = OrderedDict(sorted(msg.items(), key=lambda x: x[0]))

    convert = lambda x: '='.join([str(x[0]), ''.join(x[1])]) if type(x[1]) == list else '='.join([str(x[0]), str(x[1])]) 

    items = map(convert, s_msg.items())
    signature = '&'.join(items)   

    encrypt = hashlib.sha256()
    encrypt.update(signature.encode('utf-8'))

    sig = '{}.{}.'.format(key, tsp)
    sig += base64.b64encode(encrypt.digest()).decode('utf-8')

    print('Signing via Python to C++')

    sleep(3)
  
    return sig

def authenticate(key, secret):
    msg = {'action':'/api/v1/private/subscribe','arguments':{'instrument':['options','index'],'event':['order_book']}}
    msg['sig'] = websocket_signature(key, secret, msg['action'], msg['arguments'])
    return json.dumps(msg)
