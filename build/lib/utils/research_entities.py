import multion
from multion.client import MultiOn
multion = MultiOn(api_key="6b4973eed4df488bb139006c5c06ac4e")

def research_person(name, max_sources=4, max_steps=12, max_num_repeats=1, update_callback=None):

    prompt = (f'Your job as an expert AI agent is to help find information about a person named {name}. To do this you '+
    f'should search for "{name}" on Google. You should then find relevant results, such as Wikipedia, social media '+
    'accounts, or LinkedIn pages and summarize their contents.'+
    f"""Here are some rules I would like you to follow while you do your research:
        1. If you find multiple relevant results you should read the first one and then return back to the Google search page to read the other relevant sources.
        2. Once you run out of results that look like they have useful information about {name} please stop searching and just return your summary.
        3. You should not read more than {max_sources} results.
        4. If LinkedIn asks you to sign in please just click on the x to continue without signing in.
        5. If you need to login to a webpage just skip that webpage and move onto the next one. Do not ask me for my login credentials.
    """+
    f"""I would like you to return your information about {name} in the following format. You should follow the format exactly, including the headings. You should replace the brackets with your own content:
    Summary
    [One to two sentence summary about {name}]
    Work Experience
    [Bullet point list of {name}'s work experience. This should include his position, the company name, and the years he worked there.]
    Education
    [Bullet point list of {name}'s education. This should include schools he attended and the degrees he recieved.]
    Other Information
    [Less than 5 bullet point list of any other notable information about {name}]
    """
    )

    # Do research process
    message = research_loop(
        prompt=prompt,
        name=name,
        max_sources=max_sources,
        max_steps=max_steps,
        update_callback=update_callback
    )

    # Validate output
    num_repeats = 0
    try:
        # Check that the message fits the format given
        if (not message):
            raise Exception("Response missing message")
        if ("Summary" not in message):
            raise Exception("Missing summary")
            
        message = "Summary" + message.split("Summary")[1]
        if ("Work Experience" not in message):
            raise Exception("Missing work experience")
        if ("Education" not in message):
            raise Exception("Missing work education")
        if ("Other Information" not in message):
            raise Exception("Missing other information")
   
    except Exception as e:
        # If there's an issue with the output, rerun function up to number of repeats
        print(e)
        if num_repeats < max_num_repeats:
            research_person(
                name=name,
                max_sources=max_sources,
                max_steps=max_steps,
                max_num_repeats=max_num_repeats,
                update_callback=update_callback
            )
        else:
            raise Exception(e)
    
    return message

def research_company(name, max_sources=4, max_steps=12, max_num_repeats=1, update_callback=None):

    prompt = (f'Your job as an expert AI agent is to help find information about a company called {name}. To do this you '+
    f'should search for "{name}" on Google. You should then find relevant results, such as Wikipedia, social media '+
    "accounts, or the company's website and summarize their contents."+
    f"""Here are some rules I would like you to follow while you do your research:
        1. If you find multiple relevant results you should read the first one and then return back to the Google search page to read the other relevant sources.
        2. Once you run out of results that look like they have useful information about {name} please stop searching and just return your summary.
        3. You should not read more than five results.
        4. If a webpage asks you to sign in try to click "x" to avoid signing in.
        5. If you need to login to a webpage just skip that webpage and move onto the next one. Do not ask me for my login credentials.
    """+
    f"""I would like you to return your information about {name} in the following format. You should follow the format exactly, including the headings. You should replace the brackets with your own content:
    Summary
    [One to two sentence summary about {name}]
    Relevant Industry
    [Description of the industry {name} is in]
    Products
    [Bullet point list of any products {name} has]
    Other Information
    [Less than 5 bullet point list of any other notable information about {name}]
    """
    )

    message = research_loop(
        prompt=prompt,
        name=name,
        max_sources=max_sources,
        max_steps=max_steps,
        update_callback=update_callback
    )

    # Validate output
    num_repeats = 0
    try:
        # Check that the message fits the format given
        if (not message):
            raise Exception("Response missing message")
        if ("Summary" not in message):
            raise Exception("Missing summary")
            
        message = "Summary" + message.split("Summary")[1]
        if ("Relevant Industry" not in message):
            raise Exception("Missing Relevant Industry")
        if ("Products" not in message):
            raise Exception("Missing Products")
        if ("Other Information" not in message):
            raise Exception("Missing Other Information")
   
    except Exception as e:
        print(e)
        # If there's an issue with the output, rerun function up to number of repeats
        if num_repeats < max_num_repeats:
            research_company(
                name=name,
                max_sources=max_sources,
                max_steps=max_steps,
                max_num_repeats=max_num_repeats,
                update_callback=update_callback
            )
        else:
            raise Exception(e)
    
    return message
    


def research_loop(prompt, name, max_sources=4, max_steps=12, update_callback=None):
    """
    The research process for both researching people and companies are fundamentally the same. This function handles that process using the Multion API.
    It handles common errors and exceptions, such as the agent clicking on the same link multiple times. It also restricts the number of steps and number 
    of sources to max_steps and max_sources
    """
    # Create session
    create_session_response = multion.sessions.create(
      url="https://www.google.com/",
      local=False
    )
    session_id = create_session_response.session_id

    # Initiate loop
    i = 0
    visited_links = []
    response = None
    while i <= max_steps and (not response or response.status == 'CONTINUE'):
        i += 1
        cmd = prompt
        
        # Have agent return summary if it hasn't returned after max_steps
        if (i == max_steps):
            cmd = f'Now please return your summary of information about {name}'
            
        # If the agent has just read its nth webpage and is returning to Google have it output summary
        elif (len(visited_links) >= max_sources and response.url[:23] != "https://www.google.com/"):
            cmd = f'Now please return your summary of information about {name}'
            
        # Get link from response and check if link has already been read
        elif response and (len(visited_links) == 0 or visited_links[-1] != response.url):
            if response.url in visited_links:
                cmd = ('You have already visited this webpage. Please return to Google and find a different webpage that '+
                    'you have not visited yet. If you have visited all the relevant webpages just return your summary.')
            elif (response.url[:23] != "https://www.google.com/"):
                visited_links.append(response.url)
            
        
        # Run the multi-on api
        response = multion.sessions.step(
            session_id=session_id,
            cmd=cmd,
            url="https://www.google.com/"
        )
        update_callback({'step': i, 'max_steps': max_steps})

    if (response.status == 'NOT SURE'):
        response = multion.sessions.step(
            session_id = session_id,
            cmd="Do not take any more actions. Just return a summary of the information you have learned in the format given above.",
            url="https://www.google.com/"
        )

    # When finished end the session
    close_session_response = multion.sessions.close(session_id=session_id)
    
    if (not response.message):
        return None
    else:
        return response.message

