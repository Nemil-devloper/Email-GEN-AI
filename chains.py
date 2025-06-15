import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

load_dotenv()

class APIKeyError(Exception):
    """Custom exception for invalid or missing API keys."""
    pass

class Chain:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "your_valid_api_key_here":
            raise APIKeyError("Invalid or missing GROQ_API_KEY. Please update the .env file with a valid API key.")
        
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"  # Updated model name
        )

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template("""
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            
            ### INSTRUCTION:
            The scraped text is from a career page of a job listing.
            Please extract job titles along with available details like role, required experience, skills, and job description. 
            If some information is missing, provide an empty field for that key 
            (e.g., "experience": "", "skills": [], "description": ""). 
            Ensure the extracted information is structured in valid JSON format like this:
            [
                {{
                    "role": "Job Title",
                    "experience": "Experience",
                    "skills": ["skill1", "skill2"],
                    "description": "Job description"
                }}
            ]
            Return only valid JSON (no extra text).
        """)

        chain_extract = prompt_extract | self.llm

        try:
            res = chain_extract.invoke(input={"page_data": cleaned_text})
            json_parser = JsonOutputParser()
            parsed_result = json_parser.parse(res.content)
            return parsed_result if isinstance(parsed_result, list) else [parsed_result]
        except OutputParserException as e:
            raise OutputParserException(f"Error parsing jobs: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error during job extraction: {e}")

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template("""
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Nemil Panchal, a final-year B.E. Information Technology student at Government Engineering College, Modasa (2022–2026). 
            You have strong interests and skills in AI/ML, web development, game development. 
            You've worked on several projects, including:

            - Mathix AI – An AI-driven mathematics learning platform for students  
            - AI-based Agriculture Assistant – A web app for detecting crop diseases from images  
            - Blockchain-enabled Natural Farming Marketplace – A transparent system for farmer-consumer interaction  
            - Cricket Match Predictor – A machine learning model using Random Forest  
            - Voice-based Math Solver and a retro-style FPS game in Python

            Your technical stack includes Python, C/C++, Java, JavaScript, React, React Native, Django, Node.js, MongoDB, MySQL, Redis, 
            Git/GitHub, CI/CD, and blockchain tools like Solidity, Truffle, MetaMask, and IPFS. You're passionate about building impactful 
            tech solutions and continuously expanding your knowledge in AI/ML and full-stack development. 

            Do not provide a preamble.

            ### EMAIL (NO PREAMBLE):
        """)

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
