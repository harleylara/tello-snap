import asyncio
import websockets
import ast
import base64
import DetectNetConnector as decNet
import json
import time
import jetson.inference
import jetson.utils
import cv2


connector = decNet.DetectNetConnector()
net = jetson.inference.detectNet("ssd-mobilenet-v2")



async def process(websocket,path):
	while not websocket.closed:
		count = 0
		async for message in websocket:
	
			st = time.time()
			
			_, img_encoded = message.split(',')

			img_decoded = base64.b64decode(img_encoded)
			file_name = 'myImage.jpg'


			with open (file_name , 'wb') as f:
				f.write(img_decoded)

			image = jetson.utils.loadImage('myImage.jpg')
			detections = connector.RunInference(image,net)
			frame_2=jetson.utils.cudaToNumpy(image)
			retval, buffer = cv2.imencode('.jpg', frame_2)
			jpg_as_text = base64.b64encode(buffer)
			
			response = {"detections":detections, "image":str(jpg_as_text.decode('utf-8'))}
			response = json.dumps(response)
			

			await websocket.send(response)
			et = time.time()
			el = et-st
			print("time", el, "sec")
			count = count + 1 
	
	print("something went wrong")

async def main ():
	async with websockets.serve(process, "0.0.0.0", 4040):
		await asyncio.Future()

loop = asyncio.get_event_loop()
loop.run_until_complete(main())

