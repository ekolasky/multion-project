import unittest
import asyncio
import nest_asyncio
import websockets
import time
import json
import base64
from PIL import Image
import io
import requests


class APITests(unittest.TestCase):
    
    def test_base_endpoint(self):
        response = requests.get("http://0.0.0.0:3000")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the API"}


    # Test basic webhooks functions
    def test_connect(self):
        """
        Test connect and see if server automatically disconnects after 10 seconds
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"
            async with websockets.connect(uri) as websocket:
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                        # await websocket.close(code=1000, reason="Client initiated disconnect")
                        
                    except websockets.ConnectionClosed as e:
                        break

        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(call_webhook())
        end_time = time.time()

        # Test responses
        self.assertEqual(responses, ["Disconnected"])
        if not (8 <= (end_time - start_time) <= 12):
            raise AssertionError(f"{(end_time - start_time)} not in range [{min_value}, {max_value}]")
        
    def test_timer_refresh(self):
        """
        Test timer refresh. Connect, wait 10 seconds, send request
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"
            async with websockets.connect(uri) as websocket:

                await asyncio.sleep(5)
                await websocket.send(json.dumps({
                    "test": "Test"
                }))
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                        # await websocket.close(code=1000, reason="Client initiated disconnect")
                        
                    except websockets.ConnectionClosed as e:
                        break

        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(call_webhook())
        end_time = time.time()

        # Test responses
        self.assertEqual(responses, ["Invalid task", "Disconnected"])
        if not (14 <= (end_time - start_time) <= 16):
            raise AssertionError(f"{(end_time - start_time)} not in range [{min_value}, {max_value}]")
            

    def test_timer_no_refresh(self):
        """
        Test that timer doesn't refresh when string is sent.
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"
            async with websockets.connect(uri) as websocket:

                await asyncio.sleep(1)
                await websocket.send("test")
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                        # await websocket.close(code=1000, reason="Client initiated disconnect")
                        
                    except websockets.ConnectionClosed as e:
                        break

        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(call_webhook())
        end_time = time.time()

        # Test responses
        self.assertEqual(responses, ["Disconnected"])
        if not ((end_time - start_time) <= 5):
            raise AssertionError(f"{(end_time - start_time)} not in range [{min_value}, {max_value}]")
    
    def test_wrong_task(self):
        """
        Test task that's not extract entities or research entities
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"
            async with websockets.connect(uri) as websocket:

                await asyncio.sleep(2)
                await websocket.send(json.dumps({"task": "test"}))
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                    except websockets.ConnectionClosed as e:
                        break

        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(call_webhook())
        end_time = time.time()

        # Test responses
        self.assertEqual(responses, ['Invalid task', 'Disconnected'])
        if not (11 <=(end_time - start_time) <= 13):
            raise AssertionError(f"{(end_time - start_time)} not in range [{min_value}, {max_value}]")
        

    def test_missing_image(self):
        """
        Test case where task is extract entities but missing image
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"
            async with websockets.connect(uri) as websocket:

                await asyncio.sleep(2)
                await websocket.send(json.dumps({"task": "extract entities"}))
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                    except websockets.ConnectionClosed as e:
                        break

        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(call_webhook())
        end_time = time.time()

        # Test responses
        self.assertEqual(responses, ['Invalid image', 'Disconnected'])
        if not (11 <=(end_time - start_time) <= 13):
            raise AssertionError(f"{(end_time - start_time)} not in range [{min_value}, {max_value}]")

    
    def test_broken_image(self):
        """
        Test case where image isn't correctly formatted
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"
            async with websockets.connect(uri) as websocket:

                await asyncio.sleep(2)
                await websocket.send(json.dumps({"task": "extract entities", "image": "testing123"}))
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                    except websockets.ConnectionClosed as e:
                        break

        start_time = time.time()
        asyncio.get_event_loop().run_until_complete(call_webhook())
        end_time = time.time()

        # Test responses
        self.assertEqual(responses, ['Invalid image', 'Disconnected'])
        if not (11 <=(end_time - start_time) <= 13):
            raise AssertionError(f"{(end_time - start_time)} not in range [{min_value}, {max_value}]")

    
    def test_blank_image(self):
        """
        Test case where image returns no entities
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"

            with open('./examples/blank.png', "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create the JSON message
            message = {
                "task": "extract entities",
                "image": image_data
            }
            
            async with websockets.connect(uri) as websocket:

                await websocket.send(json.dumps(message))
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                    except websockets.ConnectionClosed as e:
                        break

        asyncio.get_event_loop().run_until_complete(call_webhook())

        # Test responses
        self.assertEqual(responses, ['Starting extract entities', 'Extract entities finished', 'Disconnected'])

    
    def test_good_image(self):
        """
        Test case where image returns entities
        """
        responses = []
        nest_asyncio.apply()
        
        async def call_webhook():
            uri = "ws://0.0.0.0:3000/ws"

            with open('./examples/email1.png', "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Create the JSON message
            message = {
                "task": "extract entities",
                "image": image_data
            }
            
            async with websockets.connect(uri) as websocket:

                await websocket.send(json.dumps(message))
        
                while True:
                    try:
                        response = await websocket.recv()
                        responses.append(response)
                        
                    except websockets.ConnectionClosed as e:
                        break

        asyncio.get_event_loop().run_until_complete(call_webhook())

        # Test responses
        self.assertEqual(responses[0], 'Starting extract entities')
        self.assertEqual(responses[-2], 'Extract entities finished')
        self.assertEqual(responses[-1], 'Disconnected')
        for entity in responses[1:-2]:
            entity_obj = json.loads(entity)
            if ('entity' not in entity_obj):
                raise AssertionError("Missing entity key")
            if ('name' not in entity_obj['entity']):
                raise AssertionError("Missing name key")
            if ('category' not in entity_obj['entity']):
                raise AssertionError("Missing name key")
            if (entity_obj['entity']['category'] != "person" and entity_obj['entity']['category'] != "company"):
                raise AssertionError("Missing name key")
                         

if __name__ == '__main__':
    unittest.main()