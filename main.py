from typing import List
from fastapi import FastAPI, WebSocket
import base64
import io
from PIL import Image
import asyncio

from utils.vlm_model import EndpointHandler
from utils.research_entities import research_company, research_person

"""
This file includes all the code for the API endpoints. The API has two endpoints. One is a basic http get request to the root "/".
The other is a webhook that allows for running both entity extraction (from a screenshot) and using a MultiOn agent to research an entity.
"""

app = FastAPI()

# Base route
@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}

# Websocket manager
class ConnectionManager:
    def __init__(self, timeout: int = 10):
        self.active_connections: List[WebSocket] = []
        self.timers: dict = {}
        self.timeout = timeout

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.timers[websocket] = asyncio.create_task(self._start_timer(websocket))

    async def send_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        await self.refresh_timer(websocket)

    async def send_obj(self, obj, websocket: WebSocket):
        await websocket.send_json(obj)
        await self.refresh_timer(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        await websocket.send_text("Disconnected")
        self.active_connections.remove(websocket)
        if websocket in self.timers:
            self.timers[websocket].cancel()
            del self.timers[websocket]
        await websocket.close()

    async def _start_timer(self, websocket: WebSocket):
        await asyncio.sleep(self.timeout)
        await self.disconnect(websocket)

    async def refresh_timer(self, websocket: WebSocket):
        if websocket in self.timers:
            self.timers[websocket].cancel()
            del self.timers[websocket]
            self.timers[websocket] = asyncio.create_task(self._start_timer(websocket))

    async def stop_timer(self, websocket: WebSocket):
        if websocket in self.timers:
            self.timers[websocket].cancel()
            del self.timers[websocket]

    async def restart_timer(self, websocket: WebSocket):
        if websocket in self.timers:
            self.timers[websocket].cancel()
            del self.timers[websocket]
        self.timers[websocket] = asyncio.create_task(self._start_timer(websocket))
        


# Initialize the handler
handler = EndpointHandler(
    base_model_id="llava-hf/llava-v1.6-34b-hf",
    adapter_model_id="ekolasky/llava-v1.6-34b-email-entities",
    load_in_4bit=True,
)

def run_async_in_new_loop(coro):
    """
    Helper function for converting asyncronous functions to syncronous functions.
    """
    new_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(new_loop)
    new_loop.run_until_complete(coro)
    new_loop.close()


# Initialize websocket
manager = ConnectionManager()
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    The websocket recieves json objects that are task specific.
    For entity extraction the json has the format {"task": "extract entities", "image": [BINARY IMAGE DATA]}
    For entity research the json has the format {"task": "research entity", "entity": {"name": [ENTITY NAME], "category": ["person" or "company"]}
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            await manager.refresh_timer(websocket)
            
            # Extract entities from message
            if ('task' in data and data['task'] == "extract entities"):
                await manager.stop_timer(websocket)
                
                try:
                    image = Image.open(io.BytesIO(base64.b64decode(data['image'])))

                    # Return entities before model finished running
                    def stream_callback(entity):
                        run_async_in_new_loop(manager.send_obj({"entity": entity}, websocket))
                    def end_stream_callback():
                        run_async_in_new_loop(manager.send_message("Extract entities finished", websocket))

                    # Run model using await with asyncio
                    await manager.send_message("Starting extract entities", websocket)
                    await asyncio.to_thread(handler, {'inputs': {'image': image}},
                        stream_callback=stream_callback,
                        end_stream_callback=end_stream_callback
                    )
                    
                    await manager.restart_timer(websocket)

                # Handle model errors
                except (KeyError, ValueError):
                    await manager.restart_timer(websocket)
                    await manager.send_message("Invalid image", websocket)
                    continue
                    
                except Exception as e:
                    print(e)
                    await manager.restart_timer(websocket)
                    await manager.send_message("Invalid image", websocket)
                    continue

            # Research entity with MultiOn agent
            elif ('task' in data and data['task'] == "research entity"):
                try:
                    await manager.stop_timer(websocket)
                    entity = data['entity']

                    # Raise exception if missing name
                    if ('name' not in entity):
                        raise Exception("Invalid entity")
                    
                    # Research company
                    if (entity['category'] == "company"):
                        await manager.send_message("Starting research entity", websocket)
                        
                        # Send update object after each step by multion agent
                        def update_callbacks(update):
                            if isinstance(update, dict):
                                run_async_in_new_loop(manager.send_obj(update, websocket))
                            elif isinstance(update, str):
                                run_async_in_new_loop(manager.send_message(update, websocket))

                        try:
                            # Run agent
                            message = await asyncio.to_thread(research_company,
                                name=entity['name'],
                                max_sources=4,
                                max_steps=12,
                                max_num_repeats=1,
                                update_callback=update_callbacks
                            )
                            await manager.send_message(message, websocket)
                        except Exception as e:
                            # Handle errors
                            print(e)
                            await manager.send_message("Agent failed during research process", websocket)
                                                

                    # Research person
                    elif (entity['category'] == "person"):
                        await manager.send_message("Starting research entity", websocket)
                        
                        # Send update object after each step by multion agent
                        def update_callbacks(update):
                            if isinstance(update, dict):
                                run_async_in_new_loop(manager.send_obj(update, websocket))
                            elif isinstance(update, str):
                                run_async_in_new_loop(manager.send_message(update, websocket))

                        try:
                            # Run agent
                            message = await asyncio.to_thread(research_person,
                                name=entity['name'],
                                max_sources=4,
                                max_steps=12,
                                max_num_repeats=1,
                                update_callback=update_callbacks
                            )
                            await manager.send_message(message, websocket)
                        except Exception as e:
                            # Handle errors
                            print(e)
                            await manager.send_message("Agent failed during research process", websocket)
                            
                    
                    # Handle unrecognized category
                    else:
                        await manager.send_message("Invalid entity", websocket)
                        continue
                    
                    await manager.restart_timer(websocket)

                # Handle errors
                except KeyError:
                    await manager.restart_timer(websocket)
                    await manager.send_message("No entity provided", websocket)
                    continue
                except Exception as e:
                    print(e)
                    await manager.restart_timer(websocket)
                    await manager.send_message("Invalid entity", websocket)
                    continue
            
            else:
                await manager.send_message("Invalid task", websocket)
    
    # On unexpected error disconnect
    except:
        if websocket in manager.active_connections:
            await manager.disconnect(websocket)