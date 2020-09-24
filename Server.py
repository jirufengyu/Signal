# WS server example that synchronizes state across clients

import asyncio
import json
import logging
import websockets

logging.basicConfig()

#Server基本参数设置
server_address='localhost'
server_port=6789

#注册到服务器上的客户端的字典
register_clients={}



async def register(websocket):#客户端接入服务器
    print('客户端：{}接入服务器'.format(websocket.remote_address))
    register_clients[websocket.remote_address]=None #将新接入的客户端信息存储进客户端字典
    print(register_clients)


async def unregister(websocket):
    print('客户端：{}离开服务器'.format(websocket.remote_address))
    del register_clients[websocket.remote_address]
    print(register_clients)
    
async def counter(websocket, path):
    # register(websocket) sends user_event() to websocket
    await register(websocket)
    try:
        async for message in websocket:
            #data = json.loads(message)
            print('客户端{}发来的数据为:{}'.format(websocket.remote_address,message))
    finally:
        await unregister(websocket)


start_server = websockets.serve(counter, server_address, server_port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()