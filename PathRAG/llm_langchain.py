import numpy as np
from typing import Union, AsyncIterator
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from .config import LLMConfig, EmbeddingConfig

def get_langchain_llm(config: LLMConfig):
    if config.use_gemini:
        return ChatGoogleGenerativeAI(
            model=config.model,
            google_api_key=config.gemini_api_key,
            temperature=config.temperature,
            convert_system_message_to_human=True
        )
    elif config.use_azure:
        return AzureChatOpenAI(
            azure_deployment=config.azure_deployment,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
            temperature=config.temperature,
        )
    else:
        return ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
        )

def get_langchain_embeddings(config: EmbeddingConfig):
    if config.use_gemini:
        return GoogleGenerativeAIEmbeddings(
            model=config.model,
            google_api_key=config.gemini_api_key
        )
    elif config.use_azure:
        return AzureOpenAIEmbeddings(
            azure_deployment=config.azure_deployment,
            azure_endpoint=config.azure_endpoint,
            api_key=config.azure_api_key,
            api_version=config.azure_api_version,
        )
    else:
        return OpenAIEmbeddings(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )

async def langchain_llm_complete(
    llm, 
    prompt: str, 
    system_prompt: str = None, 
    history_messages: list = [], 
    keyword_extraction: bool = False,
    stream: bool = False,
    **kwargs
) -> Union[str, AsyncIterator[str]]:
    
    messages: list[BaseMessage] = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    for msg in history_messages:
        role = msg.get('role')
        content = msg.get('content')
        if role == 'user':
            messages.append(HumanMessage(content=content))
        elif role == 'assistant':
            messages.append(AIMessage(content=content))
        elif role == 'system':
            messages.append(SystemMessage(content=content))
            
    if prompt:
        messages.append(HumanMessage(content=prompt))
        
    if stream:
        return _stream_generator(llm, messages)
    
    response = await llm.ainvoke(messages)
    return response.content

async def _stream_generator(llm, messages):
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content

async def langchain_embedding(embeddings_model, texts: list[str]) -> np.ndarray:
    embeddings = await embeddings_model.aembed_documents(texts)
    return np.array(embeddings)
