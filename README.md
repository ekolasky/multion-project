# Ethan Kolasky's Interview Project for MultiOn

This project is an backend API designed to take a screenshot of an email, extract the names or people and companies in the email, and then use the MultiOn API to research those people and companies. Video demos are included in the video section of the readme. You can also try it yourself by folling the insturctions in the "How to Run" section of the readme.

The extracting entities (from a screenshot) phase and researching those entities are can both be called via the API's websocket. More information about the websocket can be found in the API section below. These phases are discontinuous so extracting entities doesn't automatically research those entities. Rather the user has to send a second request to the websocket specifying the entity.

## Contents
- [Video Demos](#how-to-run)
- [How to Run](#how-to-run)
- [API](#api)
- [Training Scripts](#training-scripts)
- [Testing](#testing)
- [Design Process](#design-process)
- [Next Steps](#next-steps)

## Video Demos

The first video includes me describing the API. The other two videos are silent

- Video 1: Researching the author of an email - https://youtu.be/B7VXadKeSOo
- Video 2: Researching a company mentioned in an email - https://youtu.be/d2-69Ypcrn8
- Video 3: Researching the author of a LinkedIn message - https://youtu.be/EE7CAyP8iOA

## How to Run

### Install
Install the package by running "!pip install ." in the terminal. A notebook is also provided (start_api.ipynb) if you don't have access to the terminal (say in a Jupyter notebook in RunPod).

### Start API
You can start the API by running "!uvicorn main:app --host 0.0.0.0 --port 3000" from the terminal. You can also run this line from start_api.ipynb if you don't have access to the terminal. It's important to note that the model is run on the same server as the API. So running this will probably require a GPU.

### Test API
You can test the API locally using the notebook "test_api.ipynb". This notebook has scripts for interfacing with the API and uploading images. It's definitely the easiest way of testing the app.

## API

The API runs a websocket with two requests described below. These requests are discontinuous so the user must first extract entities and then send the selected entity to research entities. The API was designed to work with a front end app (which I didn't have time to build), which is why the two steps are discontinuous.

### Extract Entities
This request type sends a screenshot to the API and the API uses a VLM to return a list of entities in that screenshot. The request should be an object with the following format:

{"task": "extract entities", "image": BINARY IMAGE DATA}

The websocket then returns "Starting extract entities" to confirm it recieved the request. It then sends each entity objects as it is generated by the model. The entity objects have the following format:

{"entity": {"name": ENTITY NAME, "category": "person" OR "company"}}

### Research Entity
This request type sends an entity to the API and has the API use a MultiOn agent to research the entity. The request is an object with the following format:

{"task": "research entity", "entity": {"name": ENTITY NAME, "category": "person" OR "company"}

The websocket returns "Starting research entity" to confirm it recieved the request. It then sends period updates with the step of the MultiOn agent. These have the following format:

{"step": CURRENT STEP, "max_steps": MAX STEPS}

When the research process is finished the agent will send the research summary back. For people the summary contains the sections: Summary, Work Experience, Education, and Other Information. For companies the summary contains the sections: Summary, Relevant Industry, Products, and Other Information. If there's an error during the research process the webhook will return one of the following messages: "Agent failed during research process", "Invalid entity", "No entity provided", "Invalid entity", "Invalid task".

## Training Scripts

The training folder contains the training scripts, the dataset, evaluations of different models, and an evaluator. I've removed the images from the images folder for privacy reasons (becuase the images are screenshots of my person emails). As such preprocess dataset will not work.

## Testing

The test functions for the api are in the tests folder in the test_api.py file. The tests can be run from the terminal or by running the jupyter notebook run_tests.ipynb.

## Design Process

### Training the VLM
By far the hardest part of this project was getting the VLM to a suitable level of performance on this task. To do this I experimented with three different models (llava-hf/llava-v1.6-mistral-7b-hf, llava-hf/llava-v1.6-vicuna-7b-hf, and llava-hf/llava-v1.6-34b-hf) in the LLaVA family. I chose to go with the LLaVA family because LLaVA seems like one of the better performing open-source VLMs, and because, given the two day deadline, I didn't have time to train and test multiple families of models. Initial experiments revealed that llava-v1.6-34b performs significantly better than the other two models extracting 72% of the labeled entities compared to 33% for llava-v1.6-mistral-7b (a list of experiments performed can be found in training_runs.xlsx in the training folder). With this in mind I initially tried to finetune llava-v1.6-mistral-7b to see if I could overcome its performance limitations. This brought it on par with the llava-v1.6-34b.
I then tried to finetune llava-v1.6-34b to see if I could get better performance. This was significantly more difficult than I expected. I ran into a wide variety of challenges, a list of which is given below:

- Prompting Issues: The prompt listed on the hugging face model card for llava-v1.6-34b was incorrect. I realized this after my finetuned version performed significantly worse than the original version.

- Memorization of the Prompt Rather than the Desired Output: Evaluating the finetuned model revealed that, despite a lower validation loss, the finetuned model was performing significantly worse than the original model. I believe one culprit with this was that the model was being finetuned on the whole prompt rather than just the desired output. To fix this I redesigned the data collator to label just the output.

- Issues with LoRA: After fixing the data collator, I realized that the validation loss was not decreasing. Decreases in the validation loss in previous experiments were a result of the model memorizing the prompt rather than generating better responses. After doing some research I came across a training script that applied LoRA to different layers in the model than I was. Implementing this improved performance and for the first time allowed the finetuned model to surpass the performance of the non-finetuned variant.

### A full list of training runs can be found in the training_runs.xslx file

## Next Steps

Here are some possible future improvements for this project:

- Building a front-end app: The front-end app could include features like being able to share photos directly from the iOS gallery to the app, so a user could seemlessly upload their screenshot.

- Increasing the speed of the model: The model is currently quite slow. Once culprite is the quantization technique which currently trades inference speed for lower memory usage. The model could be easily sped up by either using a different library (like vLLM) or by using a different quantization technique (like AWQ).

- Improving model performance: The model currently doesn't perform as well as I would like. I think the main reason is the small training set (13 examples). Labeling more examples (or finding an existing training set for a similar task) would likely improve model performance
