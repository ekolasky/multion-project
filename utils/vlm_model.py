from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
import torch
from PIL import Image
import time

def split_list(input_list, split_elements):
    """
    Helper function to split the input_list by any of the elements in the split_elements list.
    """
    result = []
    current_part = []
    for item in input_list:
        if any([elem == item for elem in split_elements]):
            if current_part:
                result.append(current_part)
                current_part = []
        else:
            current_part.append(item)
    
    result.append(current_part)
    return result


class EndpointHandler():
    """
    Handler for the VLM model. Initializes the model based on a base model and, optionally, a set of LoRA adapters.
    """
    def __init__(self, base_model_id="llava-hf/llava-v1.6-mistral-7b-hf", adapter_model_id="", load_in_4bit=False):

        self.base_model_id = base_model_id

        # Handle quantization
        quantization_config = None
        if (load_in_4bit):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        # Initialize model and processor
        self.processor = LlavaNextProcessor.from_pretrained(base_model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )

        # Add LoRA adapters
        if (adapter_model_id != ""):
            self.model = PeftModel.from_pretrained(self.model, adapter_model_id, device_map="auto")
        
        self.prompt_creator = PromptCreator(base_model_id=base_model_id)
        self.entity_callback = None
        
    def __call__(self, data, stream_callback=None, end_stream_callback=None):
        """
        Function to call the model. Allows for streaming the model output, which returns entities as they are completed rather than all entities at the 
        end of the generation.
        """

        # Initialize empty lists for the outputs
        self.current_line_tokens = []
        self.errors = []
        self.entities = []

        
        # Get inputs
        image = data['inputs'].pop("image", None)
        if (image is None):
            raise KeyError('No image provided') 
        
        # Check image is a PIL image
        if (not isinstance(image, Image.Image)):
            raise ValueError('Invalid image') 

        
        # Tokenize prompt and image and send to model
        inputs = self.processor(self.prompt_creator(), image, return_tensors="pt").to("cuda:0")

        
        # If streaming returns the output line by line
        if (stream_callback is not None):
            # Initializes logits processor for stream_callback
            self.set_entity_callback(stream_callback)
            logits_processor_list = LogitsProcessorList([
                self.logits_processor_for_streaming
            ])

            # Run model
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=200, logits_processor=logits_processor_list)

            # Call end of stream callback
            if (end_stream_callback is not None):
                end_stream_callback()

        
        # Return full list of entities and errors if not streaming
        else:
            # Call model
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=200)
            output = self.processor.decode(output[0], skip_special_tokens=True)

            # Get response
            if (self.base_model_id == "llava-hf/llava-v1.6-mistral-7b-hf"):
                output = output.split('[/INST]')[-1]
            elif (self.base_model_id == "llava-hf/llava-v1.6-34b-hf"):
                output = output.split('<|im_start|> assistant')[-1]

            # Process each line to extract entities
            lines = output.split('\n')
            for line in lines:
                line_output = self.process_output_line(line)
                if (isinstance(line_output, str)):
                    self.errors.append(line_output)
                elif (line_output is not None):
                    self.entities.append(line_output)

            return {'entities': self.entities, 'errors': self.errors}

    
    def logits_processor_for_streaming(self, input_ids, scores):
        """
        This function is fed into model.generate to process each set of output logits from model.generate.
        The function stores each line of text and, when the line is completed, extracts the entity from the line.
        """

        # Add new tokens to current_line_tokens
        self.current_line_tokens.append(torch.argmax(scores).tolist())

        # Check if current line includes more than 1 new line token
        seperated_lines = split_list(self.current_line_tokens, [self.processor.tokenizer.encode('\n')[1], self.processor.tokenizer.eos_token_id])
        if (len(seperated_lines) == 0):
            self.current_line_tokens = seperated_lines[0]
        if (len(seperated_lines) > 0):
            self.current_line_tokens = seperated_lines[-1]

        # Process output line
        if (len(seperated_lines) >= 2):
            # Decode line and send to process_output_line
            output = self.processor.tokenizer.decode(seperated_lines[-2], skip_special_tokens=True)
            output = self.process_output_line(output)
            
            # If output obj is string add to errors list
            if (isinstance(output, str)):
                self.errors.append(output)
            # If object add to entities list and send to callback
            elif (output is not None):
                self.entities.append(output)
                if (self.entity_callback is not None):
                    self.entity_callback(output)
        
        return scores

    def set_entity_callback(self, callback):
        self.entity_callback = callback
        
    def process_output_line(self, line):
        """
        Converts a line of text to an entity object, with the keys 'name' and 'category'
        """

        # Extract categories from line
        if line == '': 
            return None
        
        sections = line.split('|')
        if ("NO NAMES FOUND" in sections[0]):
            return None
        
        if (len(sections) != 2):
            return line
        
        name_section = sections[0]
        category_section = sections[1]

        # Get name
        name_section = name_section.split('Name:')
        if (len(name_section) != 2):
            return line
        name_section = name_section[1]
        name_section = name_section.strip()
        if (len(name_section) < 2):
            return line

        # Get category
        category_section = category_section.split('Category:')
        if (len(category_section) != 2):
            return line
        category_section = category_section[1]
        category_section = category_section.strip()
        if (category_section != "person" and category_section != "company"):
            return line

        return {'name': name_section, 'category': category_section}
        
        

class PromptCreator():
    """
    PromptCreator stores the prompt. It also allows the user to quickly generate a full prompt based on the chat format for a specific base model.
    The user can also add the labeled response for generating training examples.
    """
    
    def __init__(self, base_model_id):
        self.prompt = ('Return a list of all the names of companies and people in the email. If the full name of the person or '+
        'company is not given, do not return the name. Also do not return any duplicate names. Each name should be categorized '+
        'as either a company or a person. Here is an example of how I would like you to format your output:\n\n'+
        "- Name: Elon Musk | Category: person\n"+
        "- Name: SpaceX | Category: company\n"+
        "- Name: Gwen Shotwell | Category: person\n\n"+
        'Return your output in the same format. Do not include any text outside of the format. Obviously replace the names given by '+
        'the names found in the email. If you cannot find any names return "NO NAMES FOUND". Return your output below:'
        )

        # Add chat format based on base model
        if (base_model_id == "llava-hf/llava-v1.6-mistral-7b-hf"):
            self.full_prompt = f"[INST] <image>\n{self.prompt} [/INST]"
        elif (base_model_id == "llava-hf/llava-v1.6-34b-hf"):
            self.full_prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n{self.prompt}\n<image><|im_end|><|im_start|>assistant\n"

    
    def __call__(self, entities=None):
        """
        Gets the full prompt. Either including the labeled response (in the case of training) or just the instruction (for inference).
        """
        
        if (entities is not None):
            prompt = self.full_prompt
            for i in range(len(entities["name"])):
                prompt += f"\n- Name: {entities['name'][i]} | Category: {entities['category'][i]}"
            
            return prompt
        else:
            return self.full_prompt

