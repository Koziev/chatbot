import setuptools

setuptools.setup(
    name="ruchatbot",
    version="0.0.2.2",
    author="Ilya Koziev",
    author_email="inkoziev@gmail.com",
    description="Chatbot, NLU, dialogue management",
    long_description="Chatbot and answering machine implementation for Russian language conversations",
    long_description_content_type="text/plain",
    keywords="nlp machine-learning machine-learning-library bot bots "
    "conversational-agents conversational-ai chatbot"
    "chatbot-framework bot-framework, NLP, natural-language-processing",
    url="",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["ruchatbot=ruchatbot.__main__:main"]},
)
