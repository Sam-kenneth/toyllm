from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        
        
        requirements = [req.replace("\n", "") for req in requirements]

        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(
    name='toy_llm_rag_pipeline',
    version='0.0.1',
    author='Sam-kenneth',
    author_email='awesome.mr2@gmail.com',
    description='An end-to-end RAG pipeline featuring a custom fine-tuned Toy LLM for Sherlock and Austen style generation.',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)